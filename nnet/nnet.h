#ifndef NNET_H
#define NNET_H
#include "clutil/util.h"
#include "math/geometry/dynamic_matrix.h"
#include "math/nnet/architecture.h"
#include "math/nnet/layer.h"
#include "math/nnet/layer_dimensions.h"
#include "math/nnet/symbol_generator.h"
#include "math/stats/normal.h"
#include "math/symbolic/expression.h"
#include "math/symbolic/symbolic_util.h"

#include <fstream>
#include <map>
#include <memory>
#include <sstream>

namespace nnet {

typedef double Number;

// Creates a neural network symbolically. Networks are modeled with the
// nnet::Architecture struct.
class Nnet {
 public:
  struct LearningParameters {
    Number learning_rate;
    bool dynamic_learning_rate = false;
  };

  enum InitStrategy {
    Xavier = 0,
    // none init strategy is made available for testing only (to manually set
    // weights).
    NoWeightInit
  };

  enum LossFunction {
    MeanSquared = 0,
    CrossEntropy,
  };

  // TODO(sharf): create factory class since C++'s doesn't allow named
  // parameters and I want this API to be readable.
  Nnet(const Architecture& model, InitStrategy weight_initialization = Xavier,
       LossFunction loss_function = MeanSquared)
      : model_(model), loss_function_(loss_function) {
    if (!model_.VerifyArchitecture()) {
      std::cerr << "Invalid dimensions passed to Nnet(): " << model.to_string()
                << std::endl;
      std::exit(1);
    }

    CalculateInitialWeights(weight_initialization);
  }

  // Intended mostly for testing or low-level hacks. Proceed with caution.
  Architecture model() {
    return model_;
  }

  void CompileEvaluateKernelsIfRequired(cl::Device device) {
    if (evaluate_kernels_.compiled &&
        ClDevicesAreEqual(evaluate_kernels_.device, SelectDevice())) {
      return;
    }
    std::cerr
        << "Generating and compiling OpenCl Eval kernels. This takes a while"
        << " the first time..." << std::endl;
    std::vector<std::string> eval_kernel_sources;
    // TODO const this.
    for (Layer& layer : model_.layers) {
      std::cerr << ".";
      eval_kernel_sources.push_back(layer.GenerateEvaluationKernel());
    }
    evaluate_kernels_ = CompileCl(eval_kernel_sources, device);
    std::cerr << "Done!" << std::endl;
  }

  static bool ClDevicesAreEqual(const cl::Device& a, const cl::Device& b) {
    std::string aname;
    std::string bname;
    if (CL_SUCCESS != a.getInfo(CL_DEVICE_NAME, &aname)) {
      std::cerr << "Error getting device info." << std::endl;
      return false;
    }
    if (CL_SUCCESS != b.getInfo(CL_DEVICE_NAME, &bname)) {
      std::cerr << "Error getting device info." << std::endl;
      return false;
    }

    return aname == bname;
  }

  cl::Device SelectDevice() {
    // Select the default OpenCL device.
    cl::Platform platform = clutil::GetDefaultPlatform();
    std::vector<cl::Device> devices = clutil::GetPlatformDevices(platform);
    if (devices.size() == 0) {
      std::cerr << "No OpenCL Devices on this platform." << std::endl;
      std::exit(1);
    }
    return devices[0];
  }

  Matrix<Number> EvaluateCl(Matrix<Number> in) {
    std::unique_ptr<std::vector<Matrix<Number>>> _(nullptr);
    return EvaluateCl(in, _);
  }

  // (*out_layer_outputs)[i] is a column vector containing the outputs of layer
  // i. Layer outputs will only be saved if out_layer_outputs is non-null.
  // Otherwise it will be ignored.
  Matrix<Number> EvaluateCl(
      Matrix<Number> in,
      std::unique_ptr<std::vector<Matrix<Number>>>& out_layer_outputs) {
    cl::Device device = SelectDevice();
    CompileEvaluateKernelsIfRequired(device);

    cl::Context& context = std::get<0>(evaluate_kernels_.compilation_units);
    cl::Program& program = std::get<1>(evaluate_kernels_.compilation_units);

    // Create a queue (a queue of commands that the GPU will execute)
    // Assumes that all kernels compiled for same device.
    cl::CommandQueue queue(context, device);

    // Load input.
    cl::Buffer inputs(context, CL_MEM_READ_ONLY, input_size() * sizeof(Number));
    Number inputs_buf[input_size()];
    for (size_t i = 0; i < input_size(); ++i) {
      inputs_buf[i] = in.at(i, 0);
    }
    queue.enqueueWriteBuffer(inputs, CL_TRUE, 0, sizeof(Number) * input_size(),
                             inputs_buf);

    cl::Buffer outputs;

    // TODO Load invalidated layer weights. Skip ones which haven't changed.
    for (Layer& layer : model_.layers) {
      outputs = cl::Buffer(context, CL_MEM_READ_WRITE,
                           layer.GetDimensions().num_outputs * sizeof(Number));

      // Load weights. TODO optimize the shit out of this by not re-loading
      // layers if their weights haven't changed and also caching weights_buf in
      // Layer.
      // Also, transfer all weights at once outside of this for-loop.
      const size_t number_weights = layer.weights().size();
      Number weights_buf[number_weights];
      cl::Buffer weights(context, CL_MEM_READ_WRITE,
                         layer.weights().size() * sizeof(Number));
      for (size_t i = 0; i < number_weights; ++i) {
        weights_buf[i] =
            static_cast<double>(layer.env()[layer.weights()[i]].real());
      }
      queue.enqueueWriteBuffer(weights, CL_TRUE, 0,
                               sizeof(Number) * layer.weights().size(),
                               weights_buf);

      // Evaluate.
      std::string kernel_name = layer.EvaluateKernelName();
      cl::Kernel evaluate(program, kernel_name.c_str());
      evaluate.setArg(0, inputs);
      evaluate.setArg(1, weights);
      evaluate.setArg(2, outputs);
      queue.enqueueNDRangeKernel(evaluate, cl::NullRange,
                                 cl::NDRange(layer.GetDimensions().num_outputs),
                                 cl::NullRange);

      if (out_layer_outputs) {
        Number output_buf[layer.GetDimensions().num_outputs];
        queue.enqueueReadBuffer(outputs, CL_TRUE, 0, sizeof(output_buf),
                                output_buf);
        out_layer_outputs->emplace_back(layer.GetDimensions().num_outputs, 1);
        for (size_t i = 0; i < layer.GetDimensions().num_outputs; ++i) {
          out_layer_outputs->back().at(i, 0) = output_buf[i];
        }
      }

      // inputs = outputs (output of this layer is input for next layer).
      inputs = outputs;
    }

    Number output_buf[output_size()];
    queue.enqueueReadBuffer(outputs, CL_TRUE, 0, sizeof(Number) * output_size(),
                            output_buf);
    Matrix<Number> result(output_size(), 1);
    for (size_t i = 0; i < output_size(); ++i) {
      result.at(i, 0) = output_buf[i];
    }
    return result;
  }

  void PrintColumnVector(std::string label, Matrix<Number> colvec) {
    std::cerr << "{\n\tlabel: " << label << ",\n\tdata: " << colvec.to_string()
              << "\n}" << std::endl;
  }

  cl::Buffer ColumnVectorToGpuBuffer(const cl::Context& context,
                                     cl::CommandQueue* queue,
                                     Matrix<Number> colvec) {
    if (colvec.dimensions().cols != 1) {
      std::cerr << "Matrix passed to ColumnVectorToGpuBuffer() is NOT a column "
                   "vector, and has "
                << colvec.dimensions().cols << " columns." << std::endl;
      std::exit(1);
    }
    size_t num_values = colvec.dimensions().rows;
    Number value_buf[num_values];
    cl::Buffer gpu_buffer(context, CL_MEM_READ_WRITE,
                          num_values * sizeof(Number));
    for (size_t i = 0; i < num_values; ++i) {
      value_buf[i] = colvec.at(i, 0);
    }
    queue->enqueueWriteBuffer(gpu_buffer, CL_TRUE, 0,
                              sizeof(Number) * num_values, value_buf);
    return gpu_buffer;
  }

  Matrix<Number> Evaluate(Matrix<Number> in) {
    std::unique_ptr<std::vector<Matrix<Number>>> _(nullptr);
    return Evaluate(in, _);
  }

  Matrix<Number> Evaluate(
      Matrix<Number> in,
      std::unique_ptr<std::vector<Matrix<Number>>>& out_layer_outputs) {
    Matrix<Number> inputs = in;
    Matrix<Number> outputs;
    for (Layer& layer : model_.layers) {
      symbolic::Environment env = layer.env();
      for (size_t i = 0; i < inputs.dimensions().rows; ++i) {
        env[generator_.I(i)].real() = inputs.at(i, 0);
      }

      outputs = symbolic::MapBindAndEvaluate(layer.GenerateExpression(), env);
      inputs = outputs;

      if (out_layer_outputs) {
        out_layer_outputs->emplace_back(outputs);
      }
    }

    return outputs;
  }

  // Back propagation
  void Train(Matrix<Number> in, Matrix<Number> o,
             const LearningParameters& params) {
    Number learning_rate = params.learning_rate;

    // Forward pass, store each layer's outputs as a column vector in
    // layer_outputs.
    std::unique_ptr<std::vector<Matrix<Number>>> layer_outputs =
        std::make_unique<std::vector<Matrix<Number>>>();
    Matrix<Number> actual_output = Evaluate(in, layer_outputs);

    Matrix<symbolic::Expression> output_symbolic =
        GenerateOutputLayer(output_size());

    Matrix<symbolic::Expression> expected_symbolic(o.dimensions().rows, 1);
    for (size_t i = 0; i < o.dimensions().rows; ++i) {
      expected_symbolic.at(i, 0) = symbolic::Expression(o.at(i, 0));
    }
    symbolic::Expression error =
        GenerateErrorExpression(output_symbolic, expected_symbolic);

    // Simultaneously generate symbolic expressions for output gradients and
    // build environment for evaluating them.
    Matrix<symbolic::Expression> output_gradients_symbolic(output_size(), 1);
    symbolic::Environment env;
    for (size_t i = 0; i < output_size(); ++i) {
      output_gradients_symbolic.at(i, 0) =
          error.Derive(output_symbolic.at(i, 0).to_string());

      env[generator_.O(i)] = symbolic::NumericValue(actual_output.at(i, 0));
    }

    // Generate output gradients (first part of backprop).
    Matrix<Number> gradients =
        symbolic::MapBindAndEvaluate(output_gradients_symbolic, env);

    // Propagate the gradients backwards.
    // For each layer, take the current backpropagated gradients (stored in
    // variable Matrix<Number> gradients) and pass it to the weight gradient
    // kernel to calculate weight updates. Then pass it to the input gradient
    // kernel to calculate the gradient for the next layer.
    for (int i = model_.layers.size() - 1; i >= 0; --i) {
      auto& layer = model_.layers[i];
      const Matrix<Number>& layer_input =
          (i > 0) ? layer_outputs->at(i - 1) : in;

      Matrix<symbolic::Expression> input_gradients = layer.InputGradients();

      symbolic::Environment env = layer.env();
      for (size_t i = 0; i < layer_input.dimensions().rows; ++i) {
        env[generator_.I(i)].real() = layer_input.at(i, 0);
      }
      for (size_t i = 0; i < gradients.dimensions().rows; ++i) {
        env[generator_.GRADIENT(i)].real() = gradients.at(i, 0);
      }

      // Backprop layer weight updates.
      Matrix<Number> weight_gradients =
          symbolic::MapBindAndEvaluate(layer.WeightGradients(), env);
      for (size_t i = 0; i < layer.weights().size(); ++i) {
        layer.env()[layer.weights()[i]].real() -=
            learning_rate * weight_gradients.at(i, 0);
      }

      gradients = symbolic::MapBindAndEvaluate(layer.InputGradients(), env);
    }
  }

  void CompileTrainingKernelsIfRequired(cl::Device device) {
    if (training_kernels_.compiled) {
      return;
    }

    std::cerr << "Generating and compiling OpenCl Training kernels. This takes a while"
              << " the first time." << std::endl;
    std::vector<std::string> training_kernel_sources;
    // TODO(sharf): const?
    for (Layer& layer : model_.layers) {
      std::cerr << ".";
      training_kernel_sources.push_back(layer.GenerateTrainingKernels());
    }
    training_kernels_ = CompileCl(training_kernel_sources, device);
    std::cerr << "Done!" << std::endl;
  }

  void TrainCl(Matrix<Number> in, Matrix<Number> o,
               const LearningParameters& params) {
    cl::Device device = SelectDevice();

    CompileTrainingKernelsIfRequired(device);

    cl::Context& context = std::get<0>(training_kernels_.compilation_units);
    cl::Program& program = std::get<1>(training_kernels_.compilation_units);

    // Create a queue (a queue of commands that the GPU will execute)
    // Assumes that all kernels compiled for same device.
    cl::CommandQueue queue(context, device);

    // Forward pass, store each layer's outputs as a column vector in
    // layer_outputs.
    std::unique_ptr<std::vector<Matrix<Number>>> layer_outputs =
        std::make_unique<std::vector<Matrix<Number>>>();
    Matrix<Number> actual_output = EvaluateCl(in, layer_outputs);

    Matrix<symbolic::Expression> output_symbolic =
        GenerateOutputLayer(output_size());

    Matrix<symbolic::Expression> expected_symbolic(o.dimensions().rows, 1);
    for (size_t i = 0; i < o.dimensions().rows; ++i) {
      expected_symbolic.at(i, 0) = symbolic::Expression(o.at(i, 0));
    }

    symbolic::Expression error =
        GenerateErrorExpression(output_symbolic, expected_symbolic);

    // Simultaneously generate symbolic expressions for output gradients and
    // build environment for evaluating them.
    Matrix<symbolic::Expression> output_gradients_symbolic(output_size(), 1);
    symbolic::Environment env;
    for (size_t i = 0; i < output_size(); ++i) {
      output_gradients_symbolic.at(i, 0) =
          error.Derive(output_symbolic.at(i, 0).to_string());

      env[generator_.O(i)] = symbolic::NumericValue(actual_output.at(i, 0));
    }

    // Generate output gradients (first part of backprop).
    Matrix<Number> gradients =
        symbolic::MapBindAndEvaluate(output_gradients_symbolic, env);

    cl::Buffer gpu_gradients =
          ColumnVectorToGpuBuffer(context, &queue, gradients);

    // Propagate the gradients backwards.
    // For each layer, take the current backpropagated gradients (stored in
    // variable Matrix<Number> gradients) and pass it to the weight gradient
    // kernel to calculate weight updates. Then pass it to the input gradient
    // kernel to calculate the gradient for the next layer.
    for (int i = model_.layers.size() - 1; i >= 0; --i) {
      auto& layer = model_.layers[i];
      const Matrix<Number>& layer_input =
          (i > 0) ? layer_outputs->at(i - 1) : in;

      // Load weights. TODO(sharf) optimize the shit out of this by not
      // re-loading layers if their weights haven't changed and also caching
      // weights_buf in Layer.
      // Also, transfer all weights at once outside of this for-loop.
      const size_t number_weights = layer.weights().size();
      Number weights_buf[number_weights];
      cl::Buffer weights(context, CL_MEM_READ_WRITE,
                         number_weights * sizeof(Number));
      for (size_t i = 0; i < number_weights; ++i) {
        weights_buf[i] =
            static_cast<double>(layer.env()[layer.weights()[i]].real());
      }
      queue.enqueueWriteBuffer(weights, CL_TRUE, 0,
                               sizeof(Number) * number_weights, weights_buf);

      // Load layer inputs. TODO(sharf) optimize the shit out of this by
      // keeping them in GPU memory instead of passing them to CPU and then back
      // to GPU (copied to CPU in EvaluateCL and then back to GPU here).
      // Also, transfer all inputs at once outside of this for-loop.
      cl::Buffer gpu_layer_input =
          ColumnVectorToGpuBuffer(context, &queue, layer_input);

      cl::Buffer learning_rate_buff(context, CL_MEM_READ_ONLY, sizeof(Number));
      queue.enqueueWriteBuffer(learning_rate_buff, CL_TRUE, 0, sizeof(Number),
                               &params.learning_rate);

      cl::Buffer gpu_new_weights(context, CL_MEM_READ_WRITE,
                                 number_weights * sizeof(Number));

      // Backprop layer weight updates.
      std::string weight_kernel_name = layer.WeightGradientKernelName();
      cl::Kernel weight_update(program, weight_kernel_name.c_str());
      weight_update.setArg(0, gpu_layer_input);
      weight_update.setArg(1, weights);
      weight_update.setArg(2, gpu_gradients);
      weight_update.setArg(3, gpu_new_weights);
      weight_update.setArg(4, learning_rate_buff);
      queue.enqueueNDRangeKernel(weight_update, cl::NullRange,
                                 cl::NDRange(layer.weights().size()),
                                 cl::NullRange);

      // Load in weight updates.
      Number new_weights[number_weights];
      queue.enqueueReadBuffer(gpu_new_weights, CL_TRUE, 0,
                              sizeof(Number) * number_weights, new_weights);

      for (size_t i = 0; i < number_weights; ++i) {
        layer.env()[layer.weights()[i]] =
            symbolic::NumericValue(new_weights[i]);
      }

      cl::Buffer gpu_new_gradients(
          context, CL_MEM_READ_WRITE,
          sizeof(Number) * layer.GetDimensions().num_inputs);

      // Backprop gradient calculation.
      std::string input_kernel_name = layer.InputGradientKernelName();
      cl::Kernel input_update(program, input_kernel_name.c_str());
      input_update.setArg(0, gpu_layer_input);
      input_update.setArg(1, weights);
      input_update.setArg(2, gpu_gradients);
      input_update.setArg(3, gpu_new_gradients);
      queue.enqueueNDRangeKernel(input_update, cl::NullRange,
                                 cl::NDRange(layer.GetDimensions().num_inputs),
                                 cl::NullRange);

      // Use the new input gradients for the next layer backwards (the one
      // before this one, we're iterating backwards).
      gpu_gradients = gpu_new_gradients;

    }
  }

  // TODO(sharf): move loss functions to separate file.
  symbolic::Expression GenerateErrorExpression(
      const Matrix<symbolic::Expression>& actual,
      const Matrix<symbolic::Expression>& expected) const {
    switch (loss_function_) {
      case MeanSquared:
        return GenerateMseErrorExpression(actual, expected);
      case CrossEntropy:
        return GenerateCrossEntropyErrorExpression(actual, expected);
      default:
        std::cerr << "Error: Unknown loss function selected." << std::endl;
        std::exit(1);
    }
  }

  symbolic::Expression GenerateMseErrorExpression(
      const Matrix<symbolic::Expression>& actual,
      const Matrix<symbolic::Expression>& expected) const {
    if (actual.size() != expected.size()) {
      std::cerr << "Invalid expression passed to "
                   "GenerateMseErrorExpression(Matrix<symbolic::Expression>, "
                   "Matrix<symbolic::Expression>)"
                << std::endl;
      std::exit(1);
    }

    size_t n = actual.dimensions().rows;

    symbolic::Expression error;
    for (size_t row = 0; row < actual.dimensions().rows; ++row) {
      symbolic::Expression output_error =
          (expected.at(row, 0) - actual.at(row, 0));
      error = error + (output_error * output_error);
    }
    return error / n;
  }

  symbolic::Expression GenerateCrossEntropyErrorExpression(
      const Matrix<symbolic::Expression>& actual,
      const Matrix<symbolic::Expression>& expected) const {
    if (actual.size() != expected.size()) {
      std::cerr << "Invalid expression passed to "
                   "GenerateCrossEntropyErrorExpression(Matrix<symbolic::"
                   "Expression>, "
                   "Matrix<symbolic::Expression>)"
                << std::endl;
      std::exit(1);
    }

    size_t n = actual.dimensions().rows;

    symbolic::Expression error;
    for (size_t row = 0; row < actual.dimensions().rows; ++row) {
      symbolic::Expression e = expected.at(row, 0);
      symbolic::Expression a = actual.at(row, 0);
      symbolic::Expression output_error =
          (e * symbolic::Log(a)) + ((symbolic::Expression(1) - e) *
                                    symbolic::Log(symbolic::Expression(1) - a));
      error = error + output_error;
    }
    return (symbolic::Expression(-1) / n) * (error);
  }

  std::string WeightsToString() const {
    std::stringstream output;
    output << "{";
    for (auto layer : model_.layers) {
      output << layer.WeightsToString() << "," << std::endl;
    }
    output << "}";
    return output.str();
  }

 private:
  void CalculateInitialWeights(InitStrategy weight_initialization) {
    switch (weight_initialization) {
      case NoWeightInit:
        break;
      case Xavier:
        XavierInitializeWeights();
        break;
      default:
        std::cerr << "Error: Unknown initialization strategy passed to "
                     "nnet::Nnet constructor."
                  << std::endl;
        std::exit(1);
    }
  }

  void XavierInitializeWeights() {
    for (size_t layer = 0; layer < model_.layers.size(); ++layer) {
      model_.layers[layer].XavierInitializeWeights();
    }
  }

  Matrix<symbolic::Expression> GenerateOutputLayer(size_t size) const {
    return model_.layers[model_.layers.size() - 1].OutputExpression();
  }

  size_t output_size() const { return model_.output_size(); }

  size_t input_size() const { return model_.input_size(); }

  Architecture model_;

  SymbolGenerator generator_;
  LossFunction loss_function_;

  // OpenCL state variables.
  struct OpenClState {
    bool compiled = false;
    std::tuple<cl::Context, cl::Program> compilation_units;
    cl::Device device;
  };

  OpenClState CompileCl(const std::vector<std::string>& kernel_source,
                        cl::Device device) {
    OpenClState evaluate_kernel;
    evaluate_kernel.device = device;
    evaluate_kernel.compiled = true;
    evaluate_kernel.compilation_units = clutil::Compile(device, kernel_source);
    return evaluate_kernel;
  }

  OpenClState evaluate_kernels_;
  OpenClState training_kernels_;
  std::vector<OpenClState> layer_backprop_kernels_;
};

}  // namespace nnet

#endif /* NNET_H */

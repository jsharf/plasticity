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

#include <cmath>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <future>

// ASSERT for opencl calls.
#define CL_CHECK(line) do { \
  cl_int res = line; \
  if (res != CL_SUCCESS) { \
    std::cerr << "Error running line: " #line << std::endl; \
    std::cerr << "Code: " << res << std::endl; \
    std::exit(1); \
  } \
} while(0);


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

  void CompileKernelsIfRequired(cl::Device device) {
    if (opencl_.compiled &&
        ClDevicesAreEqual(opencl_.device, SelectDevice())) {
      return;
    }
    std::cout << "Generating and compiling OpenCl kernels. This takes a while"
              << " the first time..." << std::endl;
    std::vector<std::future<std::string>> kernel_futures;
    for (const Layer& layer : model_.layers) {
      std::cerr << "/";
      kernel_futures.push_back(std::async(
          std::launch::async, &Layer::GenerateEvaluationKernel, &layer));
      kernel_futures.push_back(std::async(
          std::launch::async, &Layer::GenerateTrainingKernels, &layer));
    }

    // Wait for kernels to be ready.
    std::vector<std::string> kernel_sources;
    for (auto& kernel_future : kernel_futures) {
      std::cerr << "\\";
      kernel_sources.push_back(kernel_future.get());
    }
    std::cout << "Kernels generated. Compiling..." << std::endl;
    opencl_ = CompileCl(kernel_sources, device);
    std::cout << "Done!" << std::endl;
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

  Matrix<Number> Evaluate(Matrix<Number> in) {
    std::unique_ptr<std::vector<cl::Buffer>> _(nullptr);
    return Evaluate(in, _);
  }

  // (*out_layer_outputs)[i] is a GPU buffer containing the outputs of layer i.
  // Layer outputs will only be saved if out_layer_outputs is non-null.
  // Otherwise it will be ignored.
  Matrix<Number> Evaluate(
      Matrix<Number> in,
      std::unique_ptr<std::vector<cl::Buffer>>& out_layer_outputs) {
    cl::Device device = SelectDevice();
    CompileKernelsIfRequired(device);

    cl::Context& context = std::get<0>(opencl_.compilation_units);
    cl::Program& program = std::get<1>(opencl_.compilation_units);

    // Create a queue (a queue of commands that the GPU will execute)
    // Assumes that all kernels compiled for same device.
    cl::CommandQueue& queue = opencl_.queue;

    // Load input.
    cl_int buffer_init;
    cl::Buffer inputs(context, CL_MEM_READ_ONLY, input_size() * sizeof(Number), nullptr, &buffer_init);
    if (buffer_init != CL_SUCCESS) {
      std::cerr << "Could not initialize input buffer" << std::endl;
      std::exit(1);
    }
    Number inputs_buf[input_size()];
    for (size_t i = 0; i < input_size(); ++i) {
      inputs_buf[i] = in.at(i, 0);
    }

    cl_int result = queue.enqueueWriteBuffer(inputs, CL_TRUE, 0, sizeof(Number) * input_size(),
                             inputs_buf);
    if (result != CL_SUCCESS) {
      std::cerr << "Error enqueuing input write (Eval):  " << result << std::endl;
      std::exit(1);
    }

    cl::Buffer outputs;

    // TODO Load invalidated layer weights. Skip ones which haven't changed.
    for (size_t index = 0; index < model_.layers.size(); ++index) {
      Layer& layer = model_.layers[index];
      cl_int buffer_init;
      outputs = cl::Buffer(context, CL_MEM_READ_WRITE,
                           layer.GetDimensions().num_outputs * sizeof(Number), nullptr, &buffer_init);
      if (buffer_init != CL_SUCCESS) {
        std::cerr << "Could not initialize output buffer" << std::endl;
        std::cerr << "Layer: " << layer.LayerSuffix() << std::endl;
        std::exit(1);
      }

      // Load weights. TODO optimize the shit out of this by not re-loading
      // layers if their weights haven't changed and also caching weights_buf in
      // Layer.
      // Also, transfer all weights at once outside of this for-loop.
      const size_t number_weights = layer.weight_buffer().size();
      // This is a special case... if the layer has no weights, load one anyways
      // to prevent the system from giving an error (zero-size buffers aren't
      // allowed in opencl, and we need to provide *something* to the kernel).
      std::unique_ptr<cl::Buffer> weights;
      if (number_weights > 0) {
        cl_int buffer_init_result;
        weights = std::make_unique<cl::Buffer>(context, CL_MEM_READ_WRITE,
                                               number_weights * sizeof(Number),
                                               nullptr, &buffer_init_result);
        if (buffer_init_result != CL_SUCCESS) {
          std::cerr << "Error creating weight buffer! " << buffer_init_result << std::endl;
          std::exit(1);
        }
        result = queue.enqueueWriteBuffer(*weights, CL_TRUE, 0,
                                          sizeof(Number) * number_weights,
                                          layer.weight_buffer().data());
        if (result != CL_SUCCESS) {
          std::cerr << "Error enqueuing weights (Eval):  " << result << std::endl;
          std::exit(1);
        }
      } else {
        // Make a dummy buffer that won't be touched.
        cl_int dummy_buffer_result;
        weights = std::make_unique<cl::Buffer>(context, CL_MEM_READ_WRITE, sizeof(Number), nullptr, &dummy_buffer_result);
        if (dummy_buffer_result != 0)  {
          std::cerr << "Error initializing dummy buffer for no-weight layer." << std::endl;
          std::exit(1);
        }
      }

      // Evaluate.
      std::string kernel_name = layer.EvaluateKernelName();
      cl::Kernel evaluate(program, kernel_name.c_str());
      CL_CHECK(evaluate.setArg(0, inputs));
      CL_CHECK(evaluate.setArg(1, *weights));
      CL_CHECK(evaluate.setArg(2, outputs));
      result = queue.enqueueNDRangeKernel(
          evaluate, cl::NullRange,
          cl::NDRange(layer.GetDimensions().num_outputs), cl::NullRange);
      if (result != CL_SUCCESS) {
        std::cerr << "Error enqueuing Evaluation Kernel:  " << result << std::endl;
        std::cerr << "Layer: " << layer.LayerSuffix() << std::endl;
        std::exit(1);
      }

      if (out_layer_outputs) {
        out_layer_outputs->push_back(outputs);
      }

      // inputs = outputs (output of this layer is input for next layer).
      inputs = outputs;
    }

    Number output_buf[output_size()];
    result = queue.enqueueReadBuffer(
        outputs, CL_TRUE, 0, sizeof(Number) * output_size(), output_buf);
    if (result != CL_SUCCESS) {
      std::cerr << "Failed to read new weight values from gpu. Error code: "
                << result << std::endl;
      std::exit(1);
    }
    Matrix<Number> output(output_size(), 1);
    for (size_t i = 0; i < output_size(); ++i) {
      output.at(i, 0) = output_buf[i];
    }
    return output;
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
    cl_int result = queue->enqueueWriteBuffer(
        gpu_buffer, CL_TRUE, 0, sizeof(Number) * num_values, value_buf);
    if (result != CL_SUCCESS) {
      std::cerr << "Error enqueuing write to GPU buffer:  " << result << std::endl;
      std::exit(1);
    }
    return gpu_buffer;
  }

  void Train(Matrix<Number> in, Matrix<Number> o,
             const LearningParameters& params) {
    std::unique_ptr<Matrix<Number>> _(nullptr);
    return Train(in, o, params,  _);
  }

  void Train(Matrix<Number> in, Matrix<Number> o,
             const LearningParameters& params,
             std::unique_ptr<Matrix<Number>>& input_gradients) {
    cl::Device device = SelectDevice();

    CompileKernelsIfRequired(device);

    cl::Context& context = std::get<0>(opencl_.compilation_units);
    cl::Program& program = std::get<1>(opencl_.compilation_units);

    // Create a queue (a queue of commands that the GPU will execute)
    // Assumes that all kernels compiled for same device.
    cl::CommandQueue& queue = opencl_.queue;

    // Forward pass, store each layer's outputs as a column vector in
    // layer_outputs.
    std::unique_ptr<std::vector<cl::Buffer>> layer_outputs =
        std::make_unique<std::vector<cl::Buffer>>();
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

    double error_value = error.Bind(env).Evaluate()->real();
    std::cout << "Error (loss): " << error_value << std::endl;
    if (std::isnan(error_value)) {
      std::cerr << "The error has diverged to NaN" << std::endl;
      std::cerr << "Training value input\n=========\n " << in.to_string() << std::endl;
      std::cerr << "Training value expected output\n=========\n " << o.to_string() << std::endl;
      std::cerr << "Training value actual output\n==========\n "
                << actual_output.to_string() << std::endl;
      std::cerr << "Weights\n==============\n" << WeightsToString() << std::endl;
      //int layer_index = 1;
      //for (auto& layer : *layer_outputs) {
      //  std::cout << "layer_" << layer_index++ << ":" << std::endl;
      //  std::cout << layer << std::endl;
      //}
      std::exit(1);
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
    static int iter = 0;
    iter++;
    for (int i = model_.layers.size() - 1; i >= 0; --i) {
      auto& layer = model_.layers[i];
      const cl::Buffer& gpu_layer_input =
          (i > 0) ? layer_outputs->at(i - 1) : ColumnVectorToGpuBuffer(context, &queue, in);

      // Load weights. TODO(sharf) optimize the shit out of this by not
      // re-loading layers if their weights haven't changed and also caching
      // weights_buf in Layer.
      // Also, transfer all weights at once outside of this for-loop.
      const size_t number_weights = layer.weight_buffer().size();
      std::unique_ptr<cl::Buffer> weights;
      if (number_weights > 0) {
        weights = std::make_unique<cl::Buffer>(context, CL_MEM_READ_WRITE,
                                               number_weights * sizeof(Number));
        CL_CHECK(queue.enqueueWriteBuffer(*weights, CL_TRUE, 0,
                                          sizeof(Number) * number_weights,
                                          layer.weight_buffer().data()));
      } else {
        // Make a dummy 1-weight buffer so openCL doesn't complain about null
        // argument.
        weights = std::make_unique<cl::Buffer>(context, CL_MEM_READ_WRITE, sizeof(Number));
      }

      cl::Buffer learning_rate_buff(context, CL_MEM_READ_ONLY, sizeof(Number));
      CL_CHECK(queue.enqueueWriteBuffer(learning_rate_buff, CL_TRUE, 0,
                                        sizeof(Number), &params.learning_rate));

      if (number_weights > 0) {
        cl::Buffer gpu_new_weights(context, CL_MEM_READ_WRITE,
                                   number_weights * sizeof(Number));
        // Backprop layer weight updates.
        std::string weight_kernel_name = layer.WeightGradientKernelName();
        cl::Kernel weight_update(program, weight_kernel_name.c_str());
        CL_CHECK(weight_update.setArg(0, gpu_layer_input));
        CL_CHECK(weight_update.setArg(1, *weights));
        CL_CHECK(weight_update.setArg(2, gpu_gradients));
        CL_CHECK(weight_update.setArg(3, gpu_new_weights));
        CL_CHECK(weight_update.setArg(4, learning_rate_buff));
        cl_int result = queue.enqueueNDRangeKernel(
            weight_update, cl::NullRange,
            cl::NDRange(layer.weight_buffer().size()), cl::NullRange);
        if (result != CL_SUCCESS) {
          std::cerr << "Error enqueuing kernel "
                    << layer.WeightGradientKernelName() << std::endl;
          std::exit(1);
        }

        // Load in weight updates.
        result = queue.enqueueReadBuffer(gpu_new_weights, CL_TRUE, 0,
                                         sizeof(Number) * number_weights,
                                         layer.weight_buffer().data());
        if (result != CL_SUCCESS) {
          std::cerr << "Failed to read new weight values from gpu. Error code: "
                    << result << std::endl;
          std::exit(1);
        }
      }

      // Load in gradients
      cl::Buffer gpu_new_gradients(
          context, CL_MEM_READ_WRITE,
          sizeof(Number) * layer.GetDimensions().num_inputs);

      if (layer.GetDimensions().num_inputs > 0) {

        // Backprop gradient calculation.
        std::string input_kernel_name = layer.InputGradientKernelName();
        cl::Kernel input_update(program, input_kernel_name.c_str());
        CL_CHECK(input_update.setArg(0, gpu_layer_input));
        CL_CHECK(input_update.setArg(1, *weights));
        CL_CHECK(input_update.setArg(2, gpu_gradients));
        CL_CHECK(input_update.setArg(3, gpu_new_gradients));
        cl_int result = queue.enqueueNDRangeKernel(
            input_update, cl::NullRange,
            cl::NDRange(layer.GetDimensions().num_inputs), cl::NullRange);
        if (result != CL_SUCCESS) {
          std::cerr << "Error enqueuing kernel "
                    << layer.InputGradientKernelName() << std::endl;
          std::exit(1);
        }
      }

      // Use the new input gradients for the next layer backwards (the one
      // before this one, we're iterating backwards).
      gpu_gradients = gpu_new_gradients;

    }
    if (input_gradients) {
      size_t num_inputs = in.dimensions().rows;
      input_gradients->resize(num_inputs, 1);
      // Load in final gradients.
      std::vector<Number> temp_gradients(num_inputs, 0);
      CL_CHECK(queue.enqueueReadBuffer(gpu_gradients, CL_TRUE, 0,
                                       sizeof(Number) * num_inputs,
                                       temp_gradients.data()));
      for (size_t i = 0; i < num_inputs; ++i) {
        input_gradients->at(i, 0) = temp_gradients[i];
      }
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

  // TODO(sharf): oops, I think N might be the number of examples (1 here) in
  // the batch, not the number of rows...
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

  // TODO(sharf): oops, I think N might be the number of examples (1 here) in
  // the batch, not the number of rows...
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

    symbolic::Expression error(0.0);
    for (size_t row = 0; row < actual.dimensions().rows; ++row) {
      symbolic::Expression e = expected.at(row, 0);
      symbolic::Expression a = actual.at(row, 0);
      symbolic::Expression output_error =
          (e * symbolic::Log(a)) +
          ((symbolic::Expression(1.0) - e) *
           symbolic::Log(symbolic::Expression(1.0) - a));
      error = error + output_error;
    }
    return ((symbolic::Expression(-1.0) / n) * error);
  }

  std::string WeightsToString() const {
    std::stringstream output;
    output << "{";
    for (const auto& layer : model_.layers) {
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
    cl::CommandQueue queue;
  };

  OpenClState CompileCl(const std::vector<std::string>& kernel_source,
                        cl::Device device) {
    OpenClState kernel;
    kernel.device = device;
    kernel.compiled = true;
    kernel.compilation_units = clutil::Compile(device, kernel_source);
    kernel.queue =
        cl::CommandQueue(std::get<0>(kernel.compilation_units), device);
    return kernel;
  }

  OpenClState opencl_;
};

}  // namespace nnet

#endif /* NNET_H */

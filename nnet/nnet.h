#ifndef NNET_H
#define NNET_H
#include "clutil/util.h"
#include "math/geometry/dynamic_matrix.h"
#include "math/memory/buffer.h"
#include "math/memory/cl_buffer.h"
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
    NoWeightInit,
    // This is useful for debugging, and that's basically it. Initializes all
    // weights to 1.0.
    InitToOne,
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

    CompileKernelsIfRequired();
    for (size_t i = 0; i < model_.layers.size(); ++i) {
      model_.layers[i].RegisterToNetwork(this);
    }
    CalculateInitialWeights(weight_initialization);
  }

  std::unique_ptr<memory::Buffer> MakeBuffer(size_t size) {
    return std::make_unique<memory::ClBuffer>(
        size, &opencl_.queue, &std::get<0>(opencl_.compilation_units));
  }

  std::unique_ptr<memory::Buffer> MakeBuffer(const std::vector<double>& values) {
    return std::make_unique<memory::ClBuffer>(values, &opencl_.queue, &std::get<0>(opencl_.compilation_units));
  }

  std::unique_ptr<memory::Buffer> MakeBuffer() {
    return std::make_unique<memory::ClBuffer>(
        &opencl_.queue, &std::get<0>(opencl_.compilation_units));
  }

  void RegisterBuffer(memory::ClBuffer *buffer) {
    buffer->RegisterBackend(&opencl_.queue,
                            &std::get<0>(opencl_.compilation_units));
  }

  // Intended mostly for testing or low-level hacks. Proceed with caution.
  Architecture& model() {
    for (size_t i = 0; i < model_.layers.size(); ++i) {
      model_.layers[i].weight_buffer().MoveToCpu();
    }
    return model_;
  }

  const Layer& layer(size_t layer) {
    model_.layers[layer].weight_buffer().MoveToCpu();
    return model_.layers[layer];
  }

  void CompileKernelsIfRequired() {
    if (opencl_.compiled) {
      return;
    }
    opencl_.device = SelectDevice();
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
    opencl_ = CompileCl(kernel_sources, opencl_.device);
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

  double& GetWeight(size_t layer, size_t weight_index) {
    model_.layers[layer].weight_buffer().MoveToCpu();
    return model_.layers[layer].weight_buffer()[weight_index];
  }

  Matrix<Number> Evaluate(memory::ClBuffer& in) {
    std::unique_ptr<std::vector<cl::Buffer>> _(nullptr);
    return Evaluate(in, _);
  }

  // (*out_layer_outputs)[i] is a GPU buffer containing the outputs of layer i.
  // Layer outputs will only be saved if out_layer_outputs is non-null.
  // Otherwise it will be ignored.
  memory::ClBuffer Evaluate(
      const std::unique_ptr<memory::ClBuffer>& inputs
      std::unique_ptr<std::vector<cl::Buffer>>& out_layer_outputs) {
    CompileKernelsIfRequired();

    cl::Context& context = std::get<0>(opencl_.compilation_units);

    // Create a queue (a queue of commands that the GPU will execute)
    // Assumes that all kernels compiled for same device.
    cl::CommandQueue& queue = opencl_.queue;

    memory::ClBuffer outputs = MakeBuffer();
    outputs.MoveToGpu();

    inputs->MoveToGpu();

    // Load all weights into the GPU (weights which are already in the GPU will
    // be skipped).
    for (size_t i = 0; i < model_.layers.size(); ++i) {
      model_.layers[i].weight_buffer().MoveToGpu();
    }

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

      // Evaluate.
      std::string kernel_name = layer.EvaluateKernelName();
      cl::Kernel& evaluate = CacheFetchKernel(kernel_name);
      CL_CHECK(evaluate.setArg(0, *inputs.gpu_buffer()));
      CL_CHECK(evaluate.setArg(1, *layer.weight_buffer().gpu_buffer()));
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
      *inputs.gpu_buffer() = outputs;
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

  double Error(Matrix<Number> in, Matrix<Number> o) {
    Matrix<symbolic::Expression> output_symbolic =
        GenerateOutputLayer(output_size());

    Matrix<Number> actual_output = Evaluate(in);

    // If the initial error expressions haven't been generated yet, derive and
    // generate them now.
    if (!output_gradients_symbolic_) {
      Matrix<symbolic::Expression> expected_symbolic(o.dimensions().rows, 1);
      for (size_t i = 0; i < o.dimensions().rows; ++i) {
        expected_symbolic.at(i, 0) =
            symbolic::NumericValue("E[" + std::to_string(i) + "]");
      }

      error_ = std::make_unique<symbolic::Expression>(
          GenerateErrorExpression(output_symbolic, expected_symbolic));
    }

    // Build environment for evaluating output gradients.
    symbolic::Environment env;
    for (size_t i = 0; i < output_size(); ++i) {
      env[generator_.O(i)] = symbolic::NumericValue(actual_output.at(i, 0));
      env["E[" + std::to_string(i) + "]"] = symbolic::NumericValue(o.at(i, 0));
    }

    return error_->Bind(env).Evaluate()->real();
  }

  void Train(const Matrix<Number>& in, const Matrix<Number>& o,
             const LearningParameters& params) {
    std::unique_ptr<Matrix<Number>> _(nullptr);
    return Train(in, o, params,  _);
  }

  void Train(const Matrix<Number>& in, const Matrix<Number>& o,
             const LearningParameters& params,
             std::unique_ptr<Matrix<Number>>& input_gradients) {
    cl::Device device = SelectDevice();

    CompileKernelsIfRequired();

    cl::Context& context = std::get<0>(opencl_.compilation_units);

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

    // If the initial error expressions haven't been generated yet, derive and
    // generate them now.
    if (!output_gradients_symbolic_) {
      Matrix<symbolic::Expression> expected_symbolic(o.dimensions().rows, 1);
      for (size_t i = 0; i < o.dimensions().rows; ++i) {
        expected_symbolic.at(i, 0) =
            symbolic::NumericValue("E[" + std::to_string(i) + "]");
      }

      error_ = std::make_unique<symbolic::Expression>(
          GenerateErrorExpression(output_symbolic, expected_symbolic));

      // Generate symbolic expressions for output gradients.
      output_gradients_symbolic_ = std::make_unique<Matrix<symbolic::Expression>>(output_size(), 1);
      for (size_t i = 0; i < output_size(); ++i) {
        output_gradients_symbolic_->at(i, 0) =
            error_->Derive(output_symbolic.at(i, 0).to_string());
      }
    }

    // Build environment for evaluating output gradients.
    symbolic::Environment env;
    for (size_t i = 0; i < output_size(); ++i) {
        env[generator_.O(i)] = symbolic::NumericValue(actual_output.at(i, 0));
        env["E[" + std::to_string(i) + "]"] = symbolic::NumericValue(o.at(i, 0));
    }

    double error_value = error_->Bind(env).Evaluate()->real();
    if (std::isnan(error_value)) {
      std::cerr << "The error has diverged to NaN" << std::endl;
      std::cerr << "Training value input\n=========\n " << in.to_string() << std::endl;
      std::cerr << "Training value expected output\n=========\n " << o.to_string() << std::endl;
      std::cerr << "Training value actual output\n==========\n "
                << actual_output.to_string() << std::endl;
      std::cerr << "Weights\n==============\n" << WeightsToString() << std::endl;
      std::exit(1);
    }

    // Generate output gradients (first part of backprop).
    Matrix<Number> gradients =
        symbolic::MapBindAndEvaluate(*output_gradients_symbolic_, env);

    cl::Buffer gpu_gradients =
          ColumnVectorToGpuBuffer(context, &queue, gradients);

    // Load all weights into the GPU (weights which are already in the GPU will
    // be skipped).
    for (size_t i = 0; i < model_.layers.size(); ++i) {
      model_.layers[i].weight_buffer().MoveToGpu();
    }

    // Backpropagation algorithm.
    // For each layer, take the current backpropagated gradients (stored in
    // variable Matrix<Number> gradients) and pass it to the weight gradient
    // kernel to calculate weight updates. Then pass it to the input gradient
    // kernel to calculate the gradient for the next layer.
    for (int i = model_.layers.size() - 1; i >= 0; --i) {
      auto& layer = model_.layers[i];
      const cl::Buffer& gpu_layer_input =
          (i > 0) ? layer_outputs->at(i - 1) : ColumnVectorToGpuBuffer(context, &queue, in);

      // Load in gradients
      cl::Buffer gpu_new_gradients(
          context, CL_MEM_READ_WRITE,
          sizeof(Number) * layer.GetDimensions().num_inputs);

      if (layer.GetDimensions().num_inputs > 0) {
        // Backprop gradient calculation.
        std::string input_kernel_name = layer.InputGradientKernelName();
        cl::Kernel& input_update = CacheFetchKernel(input_kernel_name);
        CL_CHECK(input_update.setArg(0, gpu_layer_input));
        CL_CHECK(input_update.setArg(1, *layer.weight_buffer().gpu_buffer()));
        CL_CHECK(input_update.setArg(2, gpu_gradients));
        CL_CHECK(input_update.setArg(3, gpu_new_gradients));
        cl_int result = queue.enqueueNDRangeKernel(
            input_update, cl::NullRange,
            cl::NDRange(layer.GetDimensions().num_inputs), cl::NullRange);
        if (result != CL_SUCCESS) {
          std::cerr << "Error enqueuing kernel "
                    << layer.InputGradientKernelName()
                    << " & error code: " << result << std::endl;
          std::exit(1);
        }
      } else {
        std::cerr
            << "Error, incorrect model config. Layer with zero inputs found: "
            << layer.LayerSuffix() << std::endl;
      }

      cl::Buffer learning_rate_buff(context, CL_MEM_READ_ONLY, sizeof(Number));
      CL_CHECK(queue.enqueueWriteBuffer(learning_rate_buff, CL_TRUE, 0,
                                        sizeof(Number), &params.learning_rate));
  
      if (layer.weight_buffer().size() > 0) {
        cl::Buffer gpu_new_weights(context, CL_MEM_READ_WRITE,
                                   layer.weight_buffer().size() * sizeof(Number));
        // Backprop layer weight updates.
        std::string weight_kernel_name = layer.WeightGradientKernelName();
        cl::Kernel& weight_update = CacheFetchKernel(weight_kernel_name);
        CL_CHECK(weight_update.setArg(0, gpu_layer_input));
        CL_CHECK(weight_update.setArg(1, *layer.weight_buffer().gpu_buffer()));
        CL_CHECK(weight_update.setArg(2, gpu_gradients));
        CL_CHECK(weight_update.setArg(3, gpu_new_weights));
        CL_CHECK(weight_update.setArg(4, learning_rate_buff));
        cl_int result = queue.enqueueNDRangeKernel(
            weight_update, cl::NullRange,
            cl::NDRange(layer.weight_buffer().size()), cl::NullRange);
        if (result != CL_SUCCESS) {
          std::cerr << "Error enqueuing kernel "
                    << layer.WeightGradientKernelName()
                    << " & error code: " << result << std::endl;
          std::exit(1);
        }
        *layer.weight_buffer().gpu_buffer() = gpu_new_weights;
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

    symbolic::Expression error;
    for (size_t row = 0; row < actual.dimensions().rows; ++row) {
      symbolic::Expression output_error =
          (expected.at(row, 0) - actual.at(row, 0));
      error = error + (output_error * output_error);
    }
    return error / 2;
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

    symbolic::Expression error(0.0);
    for (size_t row = 0; row < actual.dimensions().rows; ++row) {
      symbolic::Expression e = expected.at(row, 0);
      symbolic::Expression a = actual.at(row, 0);
      // Should be SafeLog?
      symbolic::Expression output_error = (e * symbolic::SafeLog(a));
      error = error + output_error;
    }

    // The error formula above is actually negative...
    return (symbolic::Expression(-1.0) * error);
  }

  std::string WeightsToString() {
    std::stringstream output;
    output << "{";
    for (size_t i = 0; i < model_.layers.size(); ++i) {
      output << model_.layers[i].WeightsToString();
      if (i != model_.layers.size() - 1) {
       output << ",";
      }
      output << std::endl;
    }
    output << "}";
    return output.str();
  }

 private:
  void CalculateInitialWeights(InitStrategy weight_initialization) {
    switch (weight_initialization) {
      case NoWeightInit:
        break;
      case InitToOne:
        InitializeWeights(1.0);
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

  void InitializeWeights(double value) {
    for (size_t layer = 0; layer < model_.layers.size(); ++layer) {
      model_.layers[layer].InitializeWeights(value);
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

  // Generated error gradient expressions.
  std::unique_ptr<Matrix<symbolic::Expression>> output_gradients_symbolic_;
  std::unique_ptr<symbolic::Expression> error_;

  cl::Kernel& CacheFetchKernel(const std::string& kernel_name) {
    if (opencl_.kernels.find(kernel_name) == opencl_.kernels.end()) {
      opencl_.kernels[kernel_name] = cl::Kernel(std::get<1>(opencl_.compilation_units), kernel_name.c_str());
    }
    return opencl_.kernels[kernel_name];
  }

  // OpenCL state variables.
  struct OpenClState {
    bool compiled = false;
    std::tuple<cl::Context, cl::Program> compilation_units;
    cl::Device device;
    cl::CommandQueue queue;
    std::unordered_map<std::string, cl::Kernel> kernels;
  };

  OpenClState CompileCl(const std::vector<std::string>& kernel_source,
                        const cl::Device& device) {
    OpenClState cl_state;
    cl_state.device = device;
    cl_state.compiled = true;
    cl_state.compilation_units = clutil::Compile(device, kernel_source);
    cl_state.queue =
        cl::CommandQueue(std::get<0>(cl_state.compilation_units), device);
    return cl_state;
  }

  OpenClState opencl_;
};

}  // namespace nnet

#endif /* NNET_H */

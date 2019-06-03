#ifndef NNET_H
#define NNET_H
#include "clutil/util.h"
#include "math/geometry/dynamic_matrix.h"
#include "math/memory/cl_buffer.h"
#include "math/nnet/architecture.h"
#include "math/nnet/error_layer.h"
#include "math/nnet/layer.h"
#include "math/nnet/layer_dimensions.h"
#include "math/nnet/symbol_generator.h"
#include "math/stats/normal.h"
#include "math/symbolic/expression.h"
#include "math/symbolic/symbolic_util.h"

#include <cmath>
#include <fstream>
#include <future>
#include <map>
#include <memory>
#include <sstream>

// ASSERT for opencl calls.
#define CL_CHECK(line)                                        \
  do {                                                        \
    cl_int res = line;                                        \
    if (res != CL_SUCCESS) {                                  \
      std::cerr << "Error running line: " #line << std::endl; \
      std::cerr << "Code: " << res << std::endl;              \
      std::exit(1);                                           \
    }                                                         \
  } while (0);

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

  // TODO(sharf): create factory class since C++'s doesn't allow named
  // parameters and I want this API to be readable.
  Nnet(const Architecture &model, InitStrategy weight_initialization = Xavier,
       LossFunction loss_function = MeanSquared)
      : model_(model), error_(loss_function, model.output_size()) {
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

  std::unique_ptr<memory::ClBuffer> MakeBuffer(size_t size) {
    return std::make_unique<memory::ClBuffer>(
        size, &opencl_.queue, &std::get<0>(opencl_.compilation_units));
  }

  std::unique_ptr<memory::ClBuffer> MakeBuffer(
      const std::vector<double> &values) {
    return std::make_unique<memory::ClBuffer>(
        values, &opencl_.queue, &std::get<0>(opencl_.compilation_units));
  }

  std::unique_ptr<memory::ClBuffer> MakeBuffer() {
    return std::make_unique<memory::ClBuffer>(
        &opencl_.queue, &std::get<0>(opencl_.compilation_units));
  }

  void RegisterBuffer(memory::ClBuffer *buffer) {
    buffer->RegisterClBackend(&opencl_.queue,
                              &std::get<0>(opencl_.compilation_units));
  }

  // Intended mostly for testing or low-level hacks. Proceed with caution.
  Architecture &model() {
    for (size_t i = 0; i < model_.layers.size(); ++i) {
      model_.layers[i].weight_buffer().MoveToCpu();
    }
    return model_;
  }

  const Layer &layer(size_t layer) {
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
    for (const Layer &layer : model_.layers) {
      std::cerr << "/";
      kernel_futures.push_back(std::async(
          std::launch::async, &Layer::GenerateEvaluationKernel, &layer));
      kernel_futures.push_back(std::async(
          std::launch::async, &Layer::GenerateTrainingKernels, &layer));
    }
    kernel_futures.push_back(std::async(
        std::launch::async, &ErrorLayer::GenerateErrorKernels, &error_));

    // Wait for kernels to be ready.
    std::vector<std::string> kernel_sources;
    for (auto &kernel_future : kernel_futures) {
      std::cerr << "\\";
      kernel_sources.push_back(kernel_future.get());
    }
    std::cout << "Kernels generated. Compiling..." << std::endl;
    opencl_ = CompileCl(kernel_sources, opencl_.device);
    std::cout << "Done!" << std::endl;
  }

  static bool ClDevicesAreEqual(const cl::Device &a, const cl::Device &b) {
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

  double &GetWeight(size_t layer, size_t weight_index) {
    model_.layers[layer].weight_buffer().MoveToCpu();
    return model_.layers[layer].weight_buffer()[weight_index];
  }

  memory::ClBuffer Evaluate(memory::ClBuffer *in) {
    std::unique_ptr<std::vector<memory::ClBuffer>> _(nullptr);
    return Evaluate(in, _);
  }

  // (*out_layer_outputs)[i] is a GPU buffer containing the outputs of layer i.
  // Layer outputs will only be saved if out_layer_outputs is non-null.
  // Otherwise it will be ignored.
  memory::ClBuffer Evaluate(
      memory::ClBuffer *inputs,
      std::unique_ptr<std::vector<memory::ClBuffer>> &out_layer_outputs) {
    CompileKernelsIfRequired();

    cl::Context &context = std::get<0>(opencl_.compilation_units);

    // Create a queue (a queue of commands that the GPU will execute)
    // Assumes that all kernels compiled for same device.
    cl::CommandQueue &queue = opencl_.queue;

    std::unique_ptr<memory::ClBuffer> outputs;

    inputs->MoveToGpu();

    // Load all weights into the GPU (weights which are already in the GPU will
    // be skipped).
    for (size_t i = 0; i < model_.layers.size(); ++i) {
      model_.layers[i].weight_buffer().MoveToGpu();
    }

    for (size_t index = 0; index < model_.layers.size(); ++index) {
      Layer &layer = model_.layers[index];

      outputs = MakeBuffer(model_.layers[0].GetDimensions.num_outputs);

      // Evaluate.
      cl_int result;
      std::string kernel_name = layer.EvaluateKernelName();
      cl::Kernel &evaluate = CacheFetchKernel(kernel_name);
      CL_CHECK(evaluate.setArg(0, *inputs->gpu_buffer()));
      CL_CHECK(evaluate.setArg(1, *layer.weight_buffer().gpu_buffer()));
      CL_CHECK(evaluate.setArg(2, *outputs->gpu_buffer()));
      result = queue.enqueueNDRangeKernel(
          evaluate, cl::NullRange,
          cl::NDRange(layer.GetDimensions().num_outputs), cl::NullRange);
      if (result != CL_SUCCESS) {
        std::cerr << "Error enqueuing Evaluation Kernel:  " << result
                  << std::endl;
        std::cerr << "Layer: " << layer.LayerSuffix() << std::endl;
        std::exit(1);
      }

      if (out_layer_outputs) {
        out_layer_outputs->emplace_back(
            &opencl_.queue, &std::get<0>(opencl_.compilation_units));
        out_layer_outputs->back() = outputs;
      }

      // inputs = outputs (output of this layer is input for next layer).
      inputs = outputs.release();
    }

    return outputs;
  }

  void PrintColumnVector(std::string label, Matrix<Number> colvec) {
    std::cerr << "{\n\tlabel: " << label << ",\n\tdata: " << colvec.to_string()
              << "\n}" << std::endl;
  }

  cl::Buffer ColumnVectorToGpuBuffer(const cl::Context &context,
                                     cl::CommandQueue *queue,
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
      std::cerr << "Error enqueuing write to GPU buffer:  " << result
                << std::endl;
      std::exit(1);
    }
    return gpu_buffer;
  }

  double Error(memory::ClBuffer *actual_output, memory::ClBuffer *expected) {
    CompileKernelsIfRequired();

    actual_output->MoveToGpu();
    expected->MoveToGpu();

    memory::ClBuffer error_components = *MakeBuffer(error_.size());
    error_components.MoveToGpu();

    // Calculate error component for each output in parallel.
    cl_int result;
    std::string kernel_name = error_.ErrorKernelName();
    cl::Kernel &evaluate = CacheFetchKernel(kernel_name);
    CL_CHECK(evaluate.setArg(0, *actual_output->gpu_buffer()));
    CL_CHECK(evaluate.setArg(1, *expected->gpu_buffer()));
    CL_CHECK(evaluate.setArg(2, *error_components.gpu_buffer()));
    result = queue.enqueueNDRangeKernel(
        evaluate, cl::NullRange, cl::NDRange(error_.size()), cl::NullRange);
    if (result != CL_SUCCESS) {
      std::cerr << "Error enqueuing Error Kernel:  " << result << std::endl;
      std::exit(1);
    }

    error_components.MoveToCpu();
    double error = 0;
    for (size_t i = 0; i < error_components.size(); ++i) {
      error += error_components[i];
    }

    return error;
  }

  memory::ClBuffer ErrorGradients(memory::ClBuffer *actual_output,
                                  memory::ClBuffer *expected) {
    CompileKernelsIfRequired();

    actual_output->MoveToGpu();
    expected->MoveToGpu();

    memory::ClBuffer error_gradients = *MakeBuffer(error_.size());
    error_gradients.MoveToGpu();

    // Calculate error component for each output in parallel.
    cl_int result;
    std::string kernel_name = error_.GradientKernelName();
    cl::Kernel &evaluate = CacheFetchKernel(kernel_name);
    CL_CHECK(evaluate.setArg(0, *actual_output->gpu_buffer()));
    CL_CHECK(evaluate.setArg(1, *expected->gpu_buffer()));
    CL_CHECK(evaluate.setArg(2, *error_gradients.gpu_buffer()));
    result = queue.enqueueNDRangeKernel(
        evaluate, cl::NullRange, cl::NDRange(error_.size()), cl::NullRange);
    if (result != CL_SUCCESS) {
      std::cerr << "Error enqueuing Error Kernel:  " << result << std::endl;
      std::exit(1);
    }

    return error_gradients;
  }

  void Train(memory::ClBuffer *in, memory::ClBuffer *o,
             const LearningParameters &params) {
    std::unique_ptr<memory::ClBuffer> _(nullptr);
    return Train(in, o, params, _);
  }

  void Train(memory::ClBuffer *in, memory::ClBuffer *o,
             const LearningParameters &params,
             std::unique_ptr<memory::ClBuffer> &input_gradients) {
    cl::Device device = SelectDevice();

    CompileKernelsIfRequired();

    cl::Context &context = std::get<0>(opencl_.compilation_units);

    // Create a queue (a queue of commands that the GPU will execute)
    // Assumes that all kernels compiled for same device.
    cl::CommandQueue &queue = opencl_.queue;

    // Forward pass, store each layer's outputs as a column vector in
    // layer_outputs.
    std::unique_ptr<std::vector<memory::ClBuffer>> layer_outputs =
        std::make_unique<std::vector<memory::ClBuffer>>();
    memory::ClBuffer actual_output = Evaluate(in, layer_outputs);

    // Significant draw of GPU memory bus bandwidth. This should be removed as
    // calculating the error forces us to take data from the GPU and move it
    // back to the CPU.
    double error_value = Error(&actual_output, o);
    if (std::isnan(error_value)) {
      std::cerr << "The error has diverged to NaN" << std::endl;
      // std::cerr << "Training value input\n=========\n " << in.to_string() <<
      // std::endl;
      // std::cerr << "Training value expected output\n=========\n " <<
      // o.to_string() << std::endl;
      actual_output.MoveToCpu();
      std::cerr << "Training value actual output\n==========\n "
                << actual_output.to_string() << std::endl;
      std::cerr << "Weights\n==============\n"
                << WeightsToString() << std::endl;
      std::exit(1);
    }

    // Generate output gradients (first part of backprop).
    memory::ClBuffer gpu_gradients = ErrorGradients(&actual_output, o);

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
      auto &layer = model_.layers[i];
      const cl::Buffer &gpu_layer_input =
          (i > 0) ? layer_outputs->at(i - 1)
                  : ColumnVectorToGpuBuffer(context, &queue, in);

      // Load in gradients
      cl::Buffer gpu_new_gradients(
          context, CL_MEM_READ_WRITE,
          sizeof(Number) * layer.GetDimensions().num_inputs);

      if (layer.GetDimensions().num_inputs > 0) {
        // Backprop gradient calculation.
        std::string input_kernel_name = layer.InputGradientKernelName();
        cl::Kernel &input_update = CacheFetchKernel(input_kernel_name);
        CL_CHECK(input_update.setArg(0, gpu_layer_input));
        CL_CHECK(input_update.setArg(1, *layer.weight_buffer().gpu_buffer()));
        CL_CHECK(input_update.setArg(2, *gpu_gradients.gpu_buffer()));
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
        cl::Buffer gpu_new_weights(
            context, CL_MEM_READ_WRITE,
            layer.weight_buffer().size() * sizeof(Number));
        // Backprop layer weight updates.
        std::string weight_kernel_name = layer.WeightGradientKernelName();
        cl::Kernel &weight_update = CacheFetchKernel(weight_kernel_name);
        CL_CHECK(weight_update.setArg(0, gpu_layer_input));
        CL_CHECK(weight_update.setArg(1, *layer.weight_buffer().gpu_buffer()));
        CL_CHECK(weight_update.setArg(2, *gpu_gradients.gpu_buffer()));
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
      *input_gradients = gpu_gradients;
    }
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
  ErrorLayer error_;

  cl::Kernel &CacheFetchKernel(const std::string &kernel_name) {
    if (opencl_.kernels.find(kernel_name) == opencl_.kernels.end()) {
      opencl_.kernels[kernel_name] = cl::Kernel(
          std::get<1>(opencl_.compilation_units), kernel_name.c_str());
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

  OpenClState CompileCl(const std::vector<std::string> &kernel_source,
                        const cl::Device &device) {
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

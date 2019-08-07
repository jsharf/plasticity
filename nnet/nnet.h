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

    eval_layer_outputs_ = std::make_unique<std::vector<memory::ClBuffer>>();

    size_t max_layer_output_size = 0;
    size_t max_layer_weight_size = 0;
    for (size_t i = 0; i < model_.layers.size(); ++i) {
      model_.layers[i].RegisterToNetwork(this);

      size_t layer_size = model_.layers[i].GetDimensions().num_outputs;
      if (layer_size > max_layer_output_size) {
        max_layer_output_size = layer_size;
      }

      size_t layer_weight_size = model_.layers[i].weight_buffer().size();
      if (layer_weight_size > max_layer_weight_size) {
        max_layer_weight_size = layer_weight_size;
      }

      eval_layer_outputs_->emplace_back(
          model_.layers[i].GetDimensions().num_outputs);
      RegisterBuffer(&eval_layer_outputs_->back());
    }

    layer_outputs_ = MakeBuffer(max_layer_output_size);
    backprop_gradients_ = MakeBuffer(max_layer_output_size);
    next_backprop_gradients_ = MakeBuffer(max_layer_output_size);
    next_weight_buffer_ = MakeBuffer(max_layer_weight_size);
    learning_rate_buffer_ = MakeBuffer(1);

    layer_outputs_->MoveToGpu();
    backprop_gradients_->MoveToGpu();
    next_backprop_gradients_->MoveToGpu();
    next_weight_buffer_->MoveToGpu();
    learning_rate_buffer_->MoveToGpu();

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

  std::unique_ptr<memory::ClBuffer> MakeBuffer(
      std::initializer_list<double> values) {
    std::vector<double> vals(values);
    return MakeBuffer(vals);
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

  std::unique_ptr<memory::ClBuffer> Evaluate(const std::unique_ptr<memory::ClBuffer> &in) {
    std::unique_ptr<std::vector<memory::ClBuffer>> _(nullptr);
    return Evaluate(in, _);
  }

  // (*out_layer_outputs)[i] is a GPU buffer containing the outputs of layer i.
  // Layer outputs will only be saved if out_layer_outputs is non-null.
  // Otherwise it will be ignored.
  std::unique_ptr<memory::ClBuffer> Evaluate(
      const std::unique_ptr<memory::ClBuffer> &inputs,
      std::unique_ptr<std::vector<memory::ClBuffer>> &out_layer_outputs) {
    CompileKernelsIfRequired();

    // Create a queue (a queue of commands that the GPU will execute)
    // Assumes that all kernels compiled for same device.
    cl::CommandQueue &queue = opencl_.queue;

    std::unique_ptr<memory::ClBuffer> outputs;

    inputs->MoveToGpu();

    std::unique_ptr<memory::ClBuffer> nnet_input = std::make_unique<memory::ClBuffer>(*inputs);

    // Load all weights into the GPU (weights which are already in the GPU will
    // be skipped).
    for (size_t i = 0; i < model_.layers.size(); ++i) {
      model_.layers[i].weight_buffer().MoveToGpu();
    }

    for (size_t index = 0; index < model_.layers.size(); ++index) {
      Layer &layer = model_.layers[index];

      cl::Buffer outputs(
          std::get<0>(opencl_.compilation_units), CL_MEM_READ_WRITE,
          model_.layers[index].GetDimensions().num_outputs * sizeof(Number));

      // Evaluate.
      cl_int result;
      std::string kernel_name = layer.EvaluateKernelName();
      cl::Kernel &evaluate = CacheFetchKernel(kernel_name);
      CL_CHECK(evaluate.setArg(0, *nnet_input->gpu_buffer()));
      CL_CHECK(evaluate.setArg(1, *layer.weight_buffer().gpu_buffer()));
      CL_CHECK(evaluate.setArg(2, outputs));
      result = queue.enqueueNDRangeKernel(
          evaluate, cl::NullRange,
          cl::NDRange(layer.GetDimensions().num_outputs),
          cl::NDRange(layer.eval_workgroup_size()));
      if (result != CL_SUCCESS) {
        std::cerr << "Error enqueuing Evaluation Kernel:  " << result
                  << std::endl;
        std::cerr << "Layer: " << layer.LayerSuffix() << std::endl;
        std::exit(1);
      }

      if (out_layer_outputs) {
        out_layer_outputs->at(index).MoveToGpu();
        *out_layer_outputs->at(index).gpu_buffer() = outputs;
      }

      // inputs = outputs (output of this layer is input for next layer).
      *nnet_input->gpu_buffer() = outputs;
    }

    return std::move(nnet_input);
  }

  void PrintColumnVector(std::string label, Matrix<Number> colvec) {
    std::cerr << "{\n\tlabel: " << label << ",\n\tdata: " << colvec.to_string()
              << "\n}" << std::endl;
  }

  double Error(std::unique_ptr<memory::ClBuffer> &actual_output,
               std::unique_ptr<memory::ClBuffer> &expected) {
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
    result = opencl_.queue.enqueueNDRangeKernel(
        evaluate, cl::NullRange, cl::NDRange(error_.size()),
        cl::NDRange(error_.workgroup_size()));
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

  void ErrorGradients(
      std::unique_ptr<memory::ClBuffer> &actual_output,
      std::unique_ptr<memory::ClBuffer> &expected,
      std::unique_ptr<memory::ClBuffer> &out_error_gradients) {
    CompileKernelsIfRequired();

    actual_output->MoveToGpu();
    expected->MoveToGpu();

    // Resize the provided buffer if needed.
    if (out_error_gradients->size() < error_.size()) {
      out_error_gradients->MoveToCpu();
      out_error_gradients->resize(error_.size());
      out_error_gradients->MoveToGpu();
    }

    // Calculate error component for each output in parallel.
    cl_int result;
    std::string kernel_name = error_.GradientKernelName();
    cl::Kernel &evaluate = CacheFetchKernel(kernel_name);
    CL_CHECK(evaluate.setArg(0, *actual_output->gpu_buffer()));
    CL_CHECK(evaluate.setArg(1, *expected->gpu_buffer()));
    CL_CHECK(evaluate.setArg(2, *out_error_gradients->gpu_buffer()));
    result = opencl_.queue.enqueueNDRangeKernel(
        evaluate, cl::NullRange, cl::NDRange(error_.size()),
        cl::NDRange(error_.workgroup_size()));
    if (result != CL_SUCCESS) {
      std::cerr << "Error enqueuing Error Kernel:  " << result << std::endl;
      std::exit(1);
    }
  }

  void Train(std::unique_ptr<memory::ClBuffer> &in,
             std::unique_ptr<memory::ClBuffer> &o) {
    std::unique_ptr<memory::ClBuffer> _(nullptr);
    return Train(in, o, _);
  }

  void SetLearningParameters(const LearningParameters &params) {
    learning_rate_buffer_->MoveToCpu();
    learning_rate_buffer_->at(0) = params.learning_rate;
    learning_rate_buffer_->MoveToGpu();
  }

  void Train(std::unique_ptr<memory::ClBuffer> &in,
             std::unique_ptr<memory::ClBuffer> &o,
             const std::unique_ptr<memory::ClBuffer> &input_gradients) {
    cl::Device device = SelectDevice();

    CompileKernelsIfRequired();

    // Create a queue (a queue of commands that the GPU will execute)
    // Assumes that all kernels compiled for same device.
    cl::CommandQueue &queue = opencl_.queue;

    // Forward pass, store each layer's outputs as a column vector in
    // layer_outputs.
    std::unique_ptr<memory::ClBuffer> actual_output =
        Evaluate(in, eval_layer_outputs_);

    // Generate output gradients (first part of backprop).
    ErrorGradients(actual_output, o, backprop_gradients_);

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
      const memory::ClBuffer &gpu_layer_input =
          (i > 0) ? eval_layer_outputs_->at(i - 1) : *in;


      if (layer.GetDimensions().num_inputs > 0) {
        // Backprop gradient calculation.
        std::string input_kernel_name = layer.InputGradientKernelName();
        cl::Kernel &input_update = CacheFetchKernel(input_kernel_name);
        CL_CHECK(input_update.setArg(0, *gpu_layer_input.gpu_buffer()));
        CL_CHECK(input_update.setArg(1, *layer.weight_buffer().gpu_buffer()));
        CL_CHECK(input_update.setArg(2, *backprop_gradients_->gpu_buffer()));
        CL_CHECK(input_update.setArg(3, *next_backprop_gradients_->gpu_buffer()));
        cl_int result = queue.enqueueNDRangeKernel(
            input_update, cl::NullRange,
            cl::NDRange(layer.GetDimensions().num_inputs),
            cl::NDRange(layer.bp_train_workgroup_size()));
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

      if (layer.weight_buffer().size() > 0) {
        memory::ClBuffer gpu_new_weights = layer.weight_buffer().DeepClone();

        // Backprop layer weight updates.
        std::string weight_kernel_name = layer.WeightGradientKernelName();
        cl::Kernel &weight_update = CacheFetchKernel(weight_kernel_name);
        CL_CHECK(weight_update.setArg(0, *gpu_layer_input.gpu_buffer()));
        CL_CHECK(weight_update.setArg(1, *layer.weight_buffer().gpu_buffer()));
        CL_CHECK(weight_update.setArg(2, *backprop_gradients_->gpu_buffer()));
        CL_CHECK(weight_update.setArg(3, *gpu_new_weights.gpu_buffer()));
        CL_CHECK(weight_update.setArg(4, *learning_rate_buffer_->gpu_buffer()));
        cl_int result = queue.enqueueNDRangeKernel(
            weight_update, cl::NullRange,
            cl::NDRange(layer.weight_buffer().size()),
            cl::NDRange(layer.weight_train_workgroup_size()));
        if (result != CL_SUCCESS) {
          std::cerr << "Error enqueuing kernel "
                    << layer.WeightGradientKernelName()
                    << " & error code: " << result << std::endl;
          std::exit(1);
        }
        layer.weight_buffer() = gpu_new_weights;
      }

      // Use the new input gradients for the next layer backwards (the one
      // before this one, we're iterating backwards).
      backprop_gradients_.swap(next_backprop_gradients_);
    }
    if (input_gradients) {
      input_gradients->MoveToGpu();
      *input_gradients->gpu_buffer() = *backprop_gradients_->gpu_buffer();
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

  // Pre-allocated GPU buffers. These buffers are preallocated for performance.
  // They are used to perform neural network inference and training.
  std::unique_ptr<memory::ClBuffer> layer_outputs_;
  std::unique_ptr<memory::ClBuffer> backprop_gradients_;
  std::unique_ptr<memory::ClBuffer> next_backprop_gradients_;
  std::unique_ptr<memory::ClBuffer> next_weight_buffer_;
  std::unique_ptr<memory::ClBuffer> learning_rate_buffer_;

  std::unique_ptr<std::vector<memory::ClBuffer>> eval_layer_outputs_;

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

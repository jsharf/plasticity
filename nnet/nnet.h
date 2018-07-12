#ifndef NNET_H
#define NNET_H
#include "clutil/util.h"
#include "math/geometry/dynamic_matrix.h"
#include "math/nnet/layer.h"
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
// Nnet::Architecture struct.
class Nnet {
 public:
  struct LearningParameters {
    Number learning_rate;
    bool dynamic_learning_rate = false;
  };

  struct Architecture {
    std::vector<Layer> layers;

    bool VerifyArchitecture() const {
      // Cannot be an empty architecture.
      if (layers.size() == 0) {
        return false;
      }
      for (size_t i = 1; i < layers.size(); ++i) {
        size_t prev_output = layers[i - 1].GetDimensions().num_outputs;
        size_t curr_input = layers[i].GetDimensions().num_inputs;
        if (prev_output != curr_input) {
          return false;
        }
      }
      return true;
    }

    Architecture(size_t input_size) { AddInputLayer(input_size); }

    Architecture() {}

    // Struct in layer_impl.h.
    using Dimensions = Layer::Dimensions;

    // std::function<symbolic::Expression(const symbolic::Expression&)>
    using ActivationFunctionType = Layer::ActivationFunctionType;

    // Structs defined in convolution_layer.h.
    using VolumeDimensions = ConvolutionLayer::VolumeDimensions;
    using FilterParams = ConvolutionLayer::FilterParams;

    // Struct defined in max_pool_layer.h
    using AreaDimensions = MaxPoolLayer::AreaDimensions;

    // Input layer is just an activation layer with zero activation. Used for
    // semantics and to specify input size.
    Architecture& AddInputLayer(size_t size) {
      return AddActivationLayer(size, symbolic::Identity);
    }

    Architecture& AddFeedForwardLayer(
        size_t num_outputs, const ActivationFunctionType& activation_function) {
      Dimensions dimensions = {
          // Num inputs = num previous layer outputs.
          layers[layers.size() - 1].GetDimensions().num_outputs,
          // Num outputs specified by input parameter.
          num_outputs,
      };

      layers.push_back(Layer::MakeFeedForwardLayer(
          layers.size(), dimensions, activation_function, nullptr));
      return *this;
    }

    Architecture& AddFeedForwardLayer(size_t num_outputs) {
      Dimensions dimensions = {
          // Num inputs = num previous layer outputs.
          layers[layers.size() - 1].GetDimensions().num_outputs,
          // Num outputs specified by input parameter.
          num_outputs,
      };

      layers.push_back(
          Layer::MakeFeedForwardLayer(layers.size(), dimensions, nullptr));
      return *this;
    }

    Architecture& AddConvolutionLayer(const VolumeDimensions& dimensions,
                                      const FilterParams& params) {
      layers.push_back(Layer::MakeConvolutionLayer(layers.size(), dimensions,
                                                   params, nullptr));
      return *this;
    }

    Architecture& AddSoftmaxLayer(size_t size) {
      layers.push_back(Layer::MakeSoftmaxLayer(layers.size(), size, nullptr));
      return *this;
    }

    Architecture& AddActivationLayer(
        size_t size, const ActivationFunctionType& activation_function) {
      layers.push_back(Layer::MakeActivationLayer(
          layers.size(), size, activation_function, nullptr));
      return *this;
    }

    Architecture& AddActivationLayer(
        const ActivationFunctionType& activation_function) {
      return AddActivationLayer(
          layers[layers.size() - 1].GetDimensions().num_outputs,
          activation_function);
    }

    Architecture& AddMaxPoolLayer(const VolumeDimensions& input,
                                  const AreaDimensions& output) {
      layers.push_back(
          Layer::MakeMaxPoolLayer(layers.size(), input, output, nullptr));
      return *this;
    }

    std::string to_string() const {
      std::stringstream buffer;
      buffer << "Architecture{ inputs: " << input_size()
             << ", outputs: " << output_size() << "}";
      return buffer.str();
    }

    size_t input_size() const { return layers[0].GetDimensions().num_inputs; }
    size_t output_size() const {
      return layers[layers.size() - 1].GetDimensions().num_outputs;
    }

   private:
    friend Nnet;
    void SetSymbolGenerator(SymbolGenerator* generator) {
      for (auto& layer : layers) {
        layer.SetSymbolGenerator(generator);
      }
    };
  };

  Nnet(const Architecture& model) : model_(model) {
    if (!model_.VerifyArchitecture()) {
      std::cerr << "Invalid dimensions passed to Nnet(): " << model.to_string()
                << std::endl;
      std::exit(1);
    }

    model_.SetSymbolGenerator(&generator_);

    // "layer" is at first just a column vector of inputs.
    Matrix<symbolic::Expression> layer = GenerateInputLayer(input_size());

    std::cout << "Generating layers..." << std::endl;
    for (size_t layer_idx = 0; layer_idx < model_.layers.size(); ++layer_idx) {
      std::cout << "layers: " << layer_idx << std::endl;
      layer = model_.layers[layer_idx].GenerateExpression(layer);
    }

    // After processing all the layers, we are left with a column vector of
    // symbolic expressions. Each expression in the vector describes
    // (mathematically) the value of the output of the network W.R.T the inputs.
    neural_network_ = layer;

    CalculateInitialWeights();
  }

  // This class generates symbol names for neural network values. Since these
  // will be used for codegen for opencl, the symbols are all one-dimensional
  // indices into arrays.
  class FlatWeightSymbolGenerator : public SymbolGenerator {
   public:
    // Feed-forward layer weights.
    virtual std::string W(size_t layer, size_t node, size_t edge) {
      auto tuple = std::make_tuple(layer, node, edge);
      if (ff_weight_index_.count(tuple) == 0) {
        ff_weight_index_[tuple] = weight_count_;
        ff_rev_weight_index_[weight_count_] = tuple;
        weight_count_++;
      }
      return "W[" + std::to_string(ff_weight_index_[tuple]) + "]";
    }

    // Convolution layer weights.
    virtual std::string W(size_t layer, size_t filter, size_t x, size_t y,
                          size_t z) {
      auto tuple = std::make_tuple(layer, filter, x, y, z);
      if (conv_weight_index_.count(tuple) == 0) {
        conv_weight_index_[tuple] = weight_count_;
        conv_rev_weight_index_[weight_count_] = tuple;
        weight_count_++;
      }
      return "W[" + std::to_string(conv_weight_index_[tuple]) + "]";
    }

    // Convolution layer bias weights.
    virtual std::string W(size_t layer, size_t filter) {
      auto tuple = std::make_tuple(layer, filter);
      if (conv_bias_weight_index_.count(tuple) == 0) {
        conv_bias_weight_index_[tuple] = weight_count_;
        conv_bias_rev_weight_index_[weight_count_] = tuple;
        weight_count_++;
      }
      return "W[" + std::to_string(conv_bias_weight_index_[tuple]) + "]";
    }

    virtual std::string W(size_t i) const {
      return "W[" + std::to_string(i) + "]";
    }
    virtual std::string I(size_t i) const {
      return "I[" + std::to_string(i) + "]";
    }
    virtual std::string O(size_t i) const {
      return "O[" + std::to_string(i) + "]";
    }

    // TODO(sharf): delete this commented out code.
    //// Used to interpret results from opencl call.
    // std::map<int, std::tuple<int, int, int>> reverse_weight_map() const {
    //  return rev_weight_index_;
    //}

    size_t NumberWeights() const { return weight_count_; }

   private:
    // Mapping from <layer, node, edge> -> int. This lets each weight have a
    // single unique index.
    std::map<std::tuple<int, int, int>, int> ff_weight_index_;
    // Reverse mapping.
    std::map<int, std::tuple<int, int, int>> ff_rev_weight_index_;

    // Mapping from <layer, filter, x, y, z> -> int. This lets each weight have
    // a single unique index.
    std::map<std::tuple<int, int, int, int, int>, int> conv_weight_index_;
    // Reverse mapping.
    std::map<int, std::tuple<int, int, int, int, int>> conv_rev_weight_index_;

    // Mapping from <layer, filter> -> int. This lets each weight have a
    // single unique index.
    std::map<std::tuple<int, int>, int> conv_bias_weight_index_;
    // Reverse mapping.
    std::map<int, std::tuple<int, int>> conv_bias_rev_weight_index_;

    size_t weight_count_ = 0;
  };

  void CompileEvaluateKernelsIfRequired(cl::Device device) {
    if (evaluate_kernels_.compiled &&
        ClDevicesAreEqual(evaluate_kernels_.device, SelectDevice())) {
      return;
    }
    std::cerr << "Generating and compiling OpenCl kernel. This takes a while"
              << " the first time..." << std::endl;
    std::vector<std::string> eval_kernel_sources;
    // TODO(sharf): Shouldn't need to pass in dimensions here...
    // TODO const this.
    for (Layer& layer : model_.layers) {
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
    for (const Layer& layer : model_.layers) {
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
            static_cast<double>(weights_[layer.weights()[i]].real());
      }
      queue.enqueueWriteBuffer(weights, CL_TRUE, 0,
                               sizeof(Number) * generator_.NumberWeights(),
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

  Matrix<Number> Evaluate(Matrix<Number> in) const {
    symbolic::Environment env = weights_;
    for (size_t i = 0; i < input_size(); ++i) {
      env[generator_.I(i)].real() = in.at(i, 0);
    }

    std::function<Number(const symbolic::Expression&)> real_evaluator =
        [&env](const symbolic::Expression& exp) {
          symbolic::Expression bound_exp = exp.Bind(env);
          auto maybe_value = bound_exp.Evaluate();
          if (!maybe_value) {
            // Shit.
            std::cerr << "Well, fuck, not sure how this happened" << std::endl;
          }
          return maybe_value->real();
        };

    Matrix<Number> results = neural_network_.Map(real_evaluator);

    return results;
  }

  // Back propagation
  void Train(Matrix<Number> in, Matrix<Number> o,
             const LearningParameters& params) {
    symbolic::Environment env = weights_;
    for (size_t i = 0; i < input_size(); ++i) {
      env[generator_.I(i)].real() = in.at(i, 0);
    }
    symbolic::Environment outputs;
    for (size_t i = 0; i < output_size(); ++i) {
      env[generator_.O(i)].real() = o.at(i, 0);
    }

    symbolic::Expression error = GetErrorExpression();

    Number learning_rate = params.learning_rate;

    for (const auto& kv : weights_) {
      const std::string& weight_name = kv.first;
      symbolic::NumericValue value = kv.second;
      symbolic::Expression symbolic_gradient = error.Derive(weight_name);
      symbolic::Expression gradient = symbolic_gradient.Bind(env);
      auto gradient_value = gradient.Evaluate();
      if (!gradient_value) {
        std::cerr << "Shit" << std::endl;
        for (const std::string& variable : gradient.variables()) {
          std::cerr << variable << std::endl;
        }
        std::cerr << WeightsToString() << std::endl;
      }
      Number weight_update = -gradient_value->real() * learning_rate;
      weights_[weight_name].real() += weight_update;
    }
  }

  std::string GenerateTrainingKernelSource() {
    std::ifstream grad_descent_file(
        "math/nnet/kernels/gradient_descent.kernel.cl");
    std::stringstream buffer;
    buffer << grad_descent_file.rdbuf();
    std::string grad_descent_source = buffer.str();

    symbolic::Expression err = GetErrorExpression();
    std::stringstream gradients;
    for (size_t i = 0; i < generator_.NumberWeights(); ++i) {
      gradients << "case " << i << ":" << std::endl;
      gradients << "  return (" << err.Derive(generator_.W(i)).to_string()
                << ");" << std::endl;
    }

    std::string template_substring = "GRADIENTS_HERE";
    size_t template_location = grad_descent_source.find(template_substring);
    if (template_location == std::string::npos) {
      std::cerr << "Could not find template substring \"" << template_substring
                << "\"" << std::endl;
      std::exit(1);
    }

    grad_descent_source.replace(template_location, template_substring.size(),
                                gradients.str());
    return grad_descent_source;
  }

  void CompileTrainingKernelsIfRequired(cl::Device device) {
    if (training_kernels_.compiled) {
      return;
    }

    std::cerr << "Generating and compiling OpenCl kernel. This takes a while"
              << " the first time..." << std::endl;
    std::vector<std::string> training_kernel_sources;
    // TODO(sharf): Shouldn't need to pass in dimensions here...
    // TODO(sharf): const?
    for (Layer& layer : model_.layers) {
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
    std::unique_ptr<std::vector<Matrix<Number>>> layer_outputs;
    Matrix<Number> output = EvaluateCl(in, layer_outputs);

    Matrix<symbolic::Expression> output_symbolic =
        GenerateOutputLayer(output_size());
    Matrix<symbolic::Expression> o_symbolic(o.dimensions().rows, 1);
    for (size_t i = 0; i < o.dimensions().rows; ++i) {
      o_symbolic.at(i, 0) = symbolic::Expression(o.at(i, 0));
    }
    symbolic::Expression error =
        GenerateErrorExpression(output_symbolic, o_symbolic);

    // Simultaneously generate symbolic expressions for output gradients and
    // build environment for evaluating them.
    Matrix<symbolic::Expression> output_gradients_symbolic(output_size(), 1);
    symbolic::Environment env;
    for (size_t i = 0; i < output_size(); ++i) {
      output_gradients_symbolic.at(i, 0) =
          error.Derive(output_symbolic.at(i, 0).to_string());

      env[generator_.O(i)] = symbolic::NumericValue(output.at(i, 0));
    }

    // Generate output gradients (first part of backprop).
    Matrix<Number> gradients = output_gradients_symbolic.Map(
        std::function<Number(const symbolic::Expression&)>(
            [env](const symbolic::Expression& e) -> Number {
              symbolic::Expression bound = e.Bind(env);
              auto value = bound.Evaluate();
              if (!value) {
                std::cerr << "Unable to evaluate error gradient expression: "
                          << std::endl;
                std::cerr << e.to_string() << std::endl;
                std::exit(1);
              }
              return value->real();
            }));

    // Propagate the gradients backwards.
    // For each layer, take the current backpropagated gradients (stored in
    // variable Matrix<Number> gradients) and pass it to the weight gradient
    // kernel to calculate weight updates. Then pass it to the input gradient
    // kernel to calculate the gradient for the next layer.
    // TODO(sharf): implement this...
    for (int i = model_.layers.size() - 1; i >= 0; --i) {
      auto layer = model_.layers[i];
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
            static_cast<double>(weights_[layer.weights()[i]].real());
      }
      queue.enqueueWriteBuffer(weights, CL_TRUE, 0,
                               sizeof(Number) * number_weights, weights_buf);

      // Load layer inputs. TODO(sharf) optimize the shit out of this by
      // keeping them in GPU memory instead of passing them to CPU and then back
      // to GPU (copied to CPU in EvaluateCL and then back to GPU here).
      // Also, transfer all inputs at once outside of this for-loop.
      cl::Buffer gpu_layer_input =
          ColumnVectorToGpuBuffer(context, &queue, layer_input);

      cl::Buffer gpu_gradients =
          ColumnVectorToGpuBuffer(context, &queue, gradients);

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
        weights_[layer.weights()[i]] = symbolic::NumericValue(new_weights[i]);
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

      // Load in new gradients.
      // TODO(sharf): this loads new gradients back to CPU and then next
      // iteration puts them in GPU again. This can be optimized...
      Number new_gradients[layer.GetDimensions().num_inputs];
      queue.enqueueReadBuffer(gpu_new_gradients, CL_TRUE, 0,
                              sizeof(Number) * layer.GetDimensions().num_inputs,
                              new_gradients);
      gradients = Matrix<Number>(layer.GetDimensions().num_inputs, 1);
      for (size_t i = 0; i < layer.GetDimensions().num_inputs; ++i) {
        gradients.at(i, 0) = new_gradients[i];
      }
    }
  }

  symbolic::Expression GenerateErrorExpression(
      const Matrix<symbolic::Expression>& actual,
      const Matrix<symbolic::Expression>& expected) const {
    if (actual.size() != expected.size()) {
      std::cerr << "Invalid expression passed to "
                   "GenerateErrorExpression(Matrix<symbolic::Expression>, "
                   "Matrix<symbolic::Expression>)"
                << std::endl;
      std::exit(-1);
    }

    symbolic::Expression error;
    for (size_t row = 0; row < actual.dimensions().rows; ++row) {
      symbolic::Expression output_error =
          (expected.at(row, 0) - actual.at(row, 0));
      error = error + (output_error * output_error);
    }
    return error;
  }

  symbolic::Expression GetErrorExpression() const {
    Matrix<symbolic::Expression> expected(output_size(), 1);
    for (size_t out_idx = 0; out_idx < output_size(); ++out_idx) {
      expected.at(out_idx, 0) = symbolic::Expression(generator_.O(out_idx));
    }
    return GenerateErrorExpression(neural_network_, expected);
  }

  Matrix<symbolic::Expression> GetExpression() const { return neural_network_; }

  std::string to_string() const { return neural_network_.to_string(); }

  std::string WeightsToString() const {
    std::stringstream output;
    output << "{";
    for (const auto& kv : weights_) {
      output << kv.first << ":" << kv.second.to_string() << "," << std::endl;
    }
    output << "}";
    return output.str();
  }

 private:
  void CalculateInitialWeights() {
    for (size_t layer = 0; layer < model_.layers.size(); ++layer) {
      // Skip empty layers.
      if (model_.layers[layer].weights().size() == 0) {
        continue;
      }

      stats::Normal X = model_.layers[layer].XavierInitializer();
      for (const std::string& weight : model_.layers[layer].weights()) {
        weights_[weight].real() = X.sample();
      }
    }
  }

  Matrix<symbolic::Expression> GenerateInputLayer(size_t size) const {
    Matrix<symbolic::Expression> result(size, 1);
    for (size_t i = 0; i < size; ++i) {
      result.at(i, 0) = symbolic::CreateExpression(generator_.I(i));
    }
    return result;
  }

  Matrix<symbolic::Expression> GenerateOutputLayer(size_t size) const {
    Matrix<symbolic::Expression> result(size, 1);
    for (size_t i = 0; i < size; ++i) {
      result.at(i, 0) = symbolic::CreateExpression(generator_.O(i));
    }
    return result;
  }

  size_t output_size() const { return model_.output_size(); }

  size_t input_size() const { return model_.input_size(); }

  Architecture model_;

  // The entire neural network is stored symbolically in a column vector of
  // type symbolic::Expression. To get your outputs, simply call Bind() on all
  // expressions in the column vector with weights_ and inputs.
  // Allllso this is super inefficient. EvaluateCl and TrainCl both make use of
  // the GPU and precompile these expressions to be efficient.
  Matrix<symbolic::Expression> neural_network_;  // Dim(output_size(), 1).

  FlatWeightSymbolGenerator generator_;

  symbolic::Environment weights_;

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

#ifndef NNET_H
#define NNET_H
#include "math/geometry/dynamic_matrix.h"
#include "math/nnet/layer.h"
#include "math/stats/normal.h"
#include "math/symbolic/expression.h"
#include "math/symbolic/symbolic_util.h"
#include "clutil/util.h"

#include <fstream>
#include <map>
#include <sstream>

namespace nnet {

typedef double Number;

// Creates a neural network symbolically. The input layer has no weights. There
// are num_layers_ layers with weights.
class Nnet {
 public:
  struct LearningParameters {
    Number learning_rate;
    bool dynamic_learning_rate = false;
  };

  struct Dimensions {
    size_t num_layers;
    size_t layer_size;
    size_t output_size;
    size_t input_size;

    bool VerifySize() const {
      return (num_layers > 0) && (layer_size > 0) && (output_size > 0) &&
             (input_size > 0);
    }

    std::string to_string() const {
      std::stringstream buffer;
      buffer << "{num_layers: " << num_layers << ", layer_size: " << layer_size
             << ", output_size: " << output_size
             << ", input_size: " << input_size << "}";
      return buffer.str();
    }
  };

  Nnet(const Dimensions& dims)
      : num_layers_(dims.num_layers),
        layer_size_(dims.layer_size),
        output_size_(dims.output_size),
        input_size_(dims.input_size) {
    if (!dims.VerifySize()) {
      std::cerr << "Invalid dimensions passed to Nnet(): " << dims.to_string()
                << std::endl;
      std::exit(1);
    }

    // "layer" is at first just a column vector of inputs.
    Matrix<symbolic::Expression> layer = GenInputLayer();
    // Iterate through all layers except the output layer. The output layer's
    // dimension must match the number of outputs.
    for (size_t layer_idx = 0; layer_idx < num_layers_ - 1; ++layer_idx) {
      FeedForwardLayer::Dimensions dims;
      dims.num_outputs = layer_size_;
      dims.num_inputs = (layer_idx == 0) ? input_size_ : layer_size_;
      FeedForwardLayer layer_generator(dims, &generator_, layer_idx);
      layer = layer_generator.GenerateExpression(layer);
    }

    FeedForwardLayer::Dimensions output_dims;
    output_dims.num_inputs = (num_layers_ == 1) ? input_size_ : layer_size_;
    output_dims.num_outputs = output_size_;
    FeedForwardLayer output_layer_generator(output_dims, &generator_,
                                            num_layers_ - 1);
    neural_network_ = output_layer_generator.GenerateExpression(layer);

    CalculateInitialWeights();
  }

  // This class generates symbol names for neural network values. Since these
  // will be used for codegen for opencl, the symbols are all one-dimensional
  // indices into arrays.
  class FlatWeightSymbolGenerator : public SymbolGenerator {
   public:
    virtual std::string W(size_t layer, size_t node, size_t edge) {
      auto tuple = std::make_tuple(layer, node, edge);
      if (weight_index_.count(tuple) == 0) {
        weight_index_[tuple] = weight_count_;
        rev_weight_index_[weight_count_] = tuple;
        weight_count_++;
      }
      return "W[" + std::to_string(weight_index_[tuple]) + "]";
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

    // Used to interpret results from opencl call.
    std::map<int, std::tuple<int, int, int>> reverse_weight_map() const {
      return rev_weight_index_;
    }

    size_t NumberWeights() const { return weight_count_; }

   private:
    // Mapping from <layer, node, edge> -> int. This lets each weight have a
    // single unique index.
    std::map<std::tuple<int, int, int>, int> weight_index_;
    // Reverse mapping.
    std::map<int, std::tuple<int, int, int>> rev_weight_index_;
    size_t weight_count_ = 0;
  };

  Matrix<Number> Evaluate(Matrix<Number> in) const {
    // Modify weights_ in-place to avoid copying them. This is guaranteed to
    // never have stale inputs (from previous eval) as long as I follow naming
    // conventions and don't fuck up (I probably will).
    symbolic::Environment env = weights_;
    for (size_t i = 0; i < input_size_; ++i) {
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
  void Train(Matrix<Number> in, Matrix<Number> o, const LearningParameters& params) {
    symbolic::Environment env = weights_;
    for (size_t i = 0; i < input_size_; ++i) {
      env[generator_.I(i)].real() = in.at(i, 0);
    }
    symbolic::Environment outputs;
    for (size_t i = 0; i < output_size_; ++i) {
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
    std::ifstream grad_descent_file("kernels/gradient_descent.kernel.cl");
    std::stringstream buffer;
    buffer << grad_descent_file.rdbuf();
    std::string grad_descent_source = buffer.str();

    symbolic::Expression err = GetErrorExpression();
    std::stringstream gradients;
    for (size_t i = 0; i < generator_.NumberWeights(); ++i) {
      gradients << "case " << i << ":" << std::endl;
      gradients << "  return (" << err.Derive(generator_.W(i)).to_string() << ");"
                << std::endl;
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

  void CompileGradientDescentCl() {
    std::string kernel_source = GenerateTrainingKernelSource();
    cl::Platform platform = clutil::GetDefaultPlatform();
    std::vector<cl::Device> devices = clutil::GetPlatformDevices(platform);
    if (devices.size() == 0) {
      std::cerr << "No OpenCL Devices on this platform." << std::endl;
      std::exit(1);
    }
    device_ = devices[0];
    kernel_compiled_ = true;
    compilation_units_ = clutil::Compile(device_, {kernel_source});
  }

  void TrainCl(Matrix<Number> in, Matrix<Number> o,
               const LearningParameters& params) {
    if (!kernel_compiled_) {
      std::cout << "Generating and compiling OpenCl kernel. This takes a while"
                << " the first time..." << std::endl;
      CompileGradientDescentCl();
      std::cout << "Done!" << std::endl;
    }
    cl::Context& context = std::get<0>(compilation_units_);
    cl::Program& program = std::get<1>(compilation_units_);
    cl::Buffer new_weights(context, CL_MEM_READ_WRITE,
                           generator_.NumberWeights() * sizeof(Number));
    cl::Buffer old_weights(context, CL_MEM_READ_ONLY,
                           generator_.NumberWeights() * sizeof(Number));
    cl::Buffer inputs(context, CL_MEM_READ_ONLY, input_size_ * sizeof(Number));
    cl::Buffer outputs(context, CL_MEM_READ_ONLY, output_size_ * sizeof(Number));
    cl::Buffer learning_rate_buff(context, CL_MEM_READ_ONLY, sizeof(Number));
    Number OW[generator_.NumberWeights()];
    for (size_t i = 0; i < generator_.NumberWeights(); ++i) {
      OW[i] = static_cast<float>(weights_[generator_.W(i)].real());
    }
    Number input[input_size_];
    for (size_t i = 0; i < input_size_; ++i) {
      input[i] = in.at(i, 0);
    }
    Number output[output_size_];
    for (size_t i = 0; i < output_size_; ++i) {
      output[i] = o.at(i, 0);
    }

    // create a queue (a queue of commands that the GPU will execute)
    cl::CommandQueue queue(context, device_);

    queue.enqueueWriteBuffer(inputs, CL_TRUE, 0, sizeof(Number) * input_size_,
                             input);
    queue.enqueueWriteBuffer(outputs, CL_TRUE, 0, sizeof(Number) * output_size_,
                             output);
    queue.enqueueWriteBuffer(old_weights, CL_TRUE, 0,
                             sizeof(Number) * generator_.NumberWeights(), OW);
    queue.enqueueWriteBuffer(learning_rate_buff, CL_TRUE, 0, sizeof(Number),
                             &params.learning_rate);

    cl::Kernel gradient_descent(program, "gradient_descent");
    gradient_descent.setArg(0, inputs);
    gradient_descent.setArg(1, old_weights);
    gradient_descent.setArg(2, outputs);
    gradient_descent.setArg(3, new_weights);
    gradient_descent.setArg(4, learning_rate_buff);
    queue.enqueueNDRangeKernel(gradient_descent, cl::NullRange,
                               cl::NDRange(generator_.NumberWeights()), cl::NullRange);

    Number NW[generator_.NumberWeights()];
    queue.enqueueReadBuffer(new_weights, CL_TRUE, 0,
                            sizeof(Number) * generator_.NumberWeights(), NW);
    for (size_t i = 0; i < generator_.NumberWeights(); ++i) {
      weights_[generator_.W(i)].real() = NW[i];
    }
  }

  symbolic::Expression GetErrorExpression() const {
    symbolic::Expression error;
    for (size_t out_idx = 0; out_idx < output_size_; ++out_idx) {
      symbolic::Expression output_error =
          neural_network_.at(out_idx, 0) -
          symbolic::Expression(generator_.O(out_idx));
      error = error + (output_error * output_error);
    }
    return error;
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
    // Iterate through all layers except the output layer. The output layer's
    // dimension must match the number of outputs.
    for (size_t layer = 0; layer < num_layers_ - 1; ++layer) {
      FeedForwardLayer::Dimensions dims;
      dims.num_outputs = layer_size_;
      if (layer == 0) {
        dims.num_inputs = input_size_;
      } else {
        dims.num_inputs = layer_size_;
      }
      FeedForwardLayer layer_generator(dims, &generator_, layer);
      stats::Normal X = layer_generator.XavierInitializer();
      for (const std::string& weight : layer_generator.weights()) {
        weights_[weight].real() = X.sample();
      }
    }
    FeedForwardLayer::Dimensions output_dims;
    output_dims.num_inputs = (num_layers_ == 1) ? input_size_ : layer_size_;
    output_dims.num_outputs = output_size_;
    FeedForwardLayer output_layer_generator(output_dims, &generator_,
                                            num_layers_ - 1);
    stats::Normal O = output_layer_generator.XavierInitializer();
    for (const std::string& weight : output_layer_generator.weights()) {
      weights_[weight].real() = O.sample();
    }
  }

  Matrix<symbolic::Expression> GenInputLayer() const {
    Matrix<symbolic::Expression> result(input_size_, 1);
    for (size_t i = 0; i < input_size_; ++i) {
      result.at(i, 0) = symbolic::CreateExpression(generator_.I(i));
    }
    return result;
  }

  size_t num_layers_;
  size_t layer_size_;
  size_t output_size_;
  size_t input_size_;

  // The entire neural network is stored symbolically, in a column vector of
  // type symbolic::Expression. To get your outputs, simply call Bind() on all
  // expressions in the column vector and bind weights and inputs.
  //
  // Weight names are of the form:
  // W(i,j,k) -- i is layer number, j is the index of the neuron in that
  // layer.
  // k Is the index of the neuron in the previous layer that's connected to
  // this
  // one (or for the input layer's case, the index of the input).
  // I(i) -- i is the index of the input.
  //
  // All indices and layer numbers are zero-indexed.
  Matrix<symbolic::Expression> neural_network_;  // Dim(output_size_, 1).

  FlatWeightSymbolGenerator generator_;

  symbolic::Environment weights_;

  // OpenCL state variables.
  bool kernel_compiled_ = false;
  std::tuple<cl::Context, cl::Program> compilation_units_;
  cl::Device device_;
};

}  // namespace nnet

#endif /* NNET_H */

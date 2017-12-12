#ifndef NNET_H
#define NNET_H
#include "math/geometry/matrix.h"
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

// TODO(sharf): Things like I(), W(), weights, and evaluators should be
// encapsulated in some sort of interface.

// Creates a neural network symbolically. The input layer has no weights. There
// are kNumHiddenLayers + 2 layers with weights (+1 for output layer, +1 for
// first layer which has kInputSize number of weights per neuron).
// TODO(sharf): Implement code generation -- an expression can render itself
// into glsl for parallel execution. That will make this useful.
template <size_t kNumHiddenLayers, size_t kLayerSize, size_t kOutputSize,
          size_t kInputSize>
class Nnet {
 public:
  // + 1 for bias.
  static constexpr size_t kNumLayers = kNumHiddenLayers + 2;

  using SymbolicInputVector = Matrix<kInputSize, 1, symbolic::Expression>;
  using SymbolicOutputVector = Matrix<kOutputSize, 1, symbolic::Expression>;
  using InputVector = Matrix<kInputSize, 1, Number>;
  using OutputVector = Matrix<kOutputSize, 1, Number>;
  using HiddenVector = Matrix<kLayerSize, 1, symbolic::Expression>;

  struct LearningParameters {
    Number learning_rate;
    bool dynamic_learning_rate = false;
  };

  Nnet() {
    SymbolicInputVector inputs = GenInputLayer();
    FeedForwardLayer<kInputSize, kLayerSize> input_layer_generator(&generator_,
                                                                   0);

    HiddenVector layer = input_layer_generator.GenerateExpression(inputs);

    for (size_t layer_idx = 1; layer_idx <= kNumHiddenLayers; ++layer_idx) {
      FeedForwardLayer<kLayerSize, kLayerSize> hidden_layer_generator(
          &generator_, layer_idx);
      layer = hidden_layer_generator.GenerateExpression(layer);
    }

    FeedForwardLayer<kLayerSize, kOutputSize> output_layer_generator(
        &generator_, kNumLayers - 1);
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

  OutputVector Evaluate(InputVector in) const {
    // Modify weights_ in-place to avoid copying them. This is guaranteed to
    // never have stale inputs (from previous eval) as long as I follow naming
    // conventions and don't fuck up (I probably will).
    symbolic::Environment weights = weights_;
    for (size_t i = 0; i < kInputSize; ++i) {
      weights[generator_.I(i)].real() = in.at(i, 0);
    }

    std::function<symbolic::Expression(const symbolic::Expression&)> binder =
        [&weights](const symbolic::Expression& exp) {
          return exp.Bind(weights);
        };

    Matrix<kOutputSize, 1, symbolic::Expression> bound_network =
        neural_network_.Map(binder);

    std::function<Number(const symbolic::Expression&)> real_evaluator =
        [](const symbolic::Expression& exp) {
          auto maybe_value = exp.Evaluate();
          if (!maybe_value) {
            // Shit.
            std::cerr << "Well, fuck, not sure how this happened" << std::endl;
          }
          return maybe_value->real();
        };

    Matrix<kOutputSize, 1, Number> results = bound_network.Map(real_evaluator);

    return results;
  }

  // Back propagation
  void Train(InputVector in, OutputVector o, const LearningParameters& params) {
    for (size_t i = 0; i < kInputSize; ++i) {
      weights_[generator_.I(i)].real() = in.at(i, 0);
    }
    for (size_t i = 0; i < kOutputSize; ++i) {
      weights_[generator_.O(i)].real() = o.at(i, 0);
    }

    symbolic::Expression error = GetErrorExpression();
    symbolic::Environment weights = weights_;

    Number learning_rate = params.learning_rate;

    for (const auto& kv : weights) {
      const std::string& weight_name = kv.first;
      symbolic::NumericValue value = kv.second;
      symbolic::Expression symbolic_gradient = error.Derive(weight_name);
      symbolic::Expression gradient = symbolic_gradient.Bind(weights);
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
    gradients << "{";
    for (size_t i = 0; i < generator_.NumberWeights(); ++i) {
      gradients << err.Derive(generator_.W(i)).to_string() << "," << std::endl;
    }
    gradients << "}";

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

  void TrainCl(InputVector in, OutputVector o,
               const LearningParameters& params) {
    if (!kernel_compiled_) {
      CompileGradientDescentCl();
    }
    cl::Context& context = std::get<0>(compilation_units_);
    cl::Program& program = std::get<1>(compilation_units_);
    cl::Buffer new_weights(context, CL_MEM_READ_WRITE,
                           generator_.NumberWeights() * sizeof(Number));
    cl::Buffer old_weights(context, CL_MEM_READ_ONLY,
                           generator_.NumberWeights() * sizeof(Number));
    cl::Buffer inputs(context, CL_MEM_READ_ONLY, kInputSize * sizeof(Number));
    cl::Buffer outputs(context, CL_MEM_READ_ONLY, kOutputSize * sizeof(Number));
    cl::Buffer learning_rate_buff(context, CL_MEM_READ_ONLY, sizeof(Number));
    Number OW[generator_.NumberWeights()];
    for (size_t i = 0; i < generator_.NumberWeights(); ++i) {
      OW[i] = weights_[generator_.W(i)].real();
    }

    // create a queue (a queue of commands that the GPU will execute)
    cl::CommandQueue queue(context, device_);

    queue.enqueueWriteBuffer(inputs, CL_TRUE, 0, sizeof(Number) * kInputSize,
                             in.data());
    queue.enqueueWriteBuffer(outputs, CL_TRUE, 0, sizeof(Number) * kOutputSize,
                             o.data());
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
    for (size_t out_idx = 0; out_idx < kOutputSize; ++out_idx) {
      symbolic::Expression output_error =
          neural_network_.at(out_idx, 0) -
          symbolic::Expression(generator_.O(out_idx));
      error = error + (output_error * output_error);
    }
    return error;
  }

  SymbolicOutputVector GetExpression() const { return neural_network_; }

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
    FeedForwardLayer<kInputSize, kLayerSize> input_layer_generator(&generator_,
                                                                   0);
    stats::Normal I = input_layer_generator.XavierInitializer();
    for (const std::string& weight : input_layer_generator.weights()) {
      weights_[weight].real() = I.sample();
    }
    for (size_t layer = 1; layer <= kNumHiddenLayers; ++layer) {
      FeedForwardLayer<kLayerSize, kLayerSize> hidden_layer_generator(
          &generator_, layer);
      stats::Normal H = hidden_layer_generator.XavierInitializer();
      for (const std::string& weight : hidden_layer_generator.weights()) {
        weights_[weight].real() = H.sample();
      }
    }
    FeedForwardLayer<kLayerSize, kOutputSize> output_layer_generator(
        &generator_, kNumLayers - 1);
    stats::Normal O = output_layer_generator.XavierInitializer();
    for (const std::string& weight : output_layer_generator.weights()) {
      weights_[weight].real() = O.sample();
    }
  }

  SymbolicInputVector GenInputLayer() const {
    SymbolicInputVector result;
    for (size_t i = 0; i < kInputSize; ++i) {
      result.at(i, 0) = symbolic::CreateExpression(generator_.I(i));
    }
    return result;
  }

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
  //
  // The first layer has kInputSize nuerons. Every Hidden layer has kLayerSize
  // neurons. The output layer has kOutputSize neurons. There are
  // kNumHiddenLayers + 1 layers with weights (the hidden layers and the
  // output
  // layer).
  Matrix<kOutputSize, 1, symbolic::Expression> neural_network_;

  FlatWeightSymbolGenerator generator_;

  symbolic::Environment weights_;

  // For efficiently iterating through weights.
  std::vector<std::string> weight_list_;

  bool kernel_compiled_ = false;
  std::tuple<cl::Context, cl::Program> compilation_units_;
  cl::Device device_;
};

}  // namespace nnet

#endif /* NNET_H */

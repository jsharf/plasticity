#ifndef NNET_H
#define NNET_H
#include "math/geometry/matrix.h"
#include "math/stats/normal.h"
#include "math/symbolic/expression.h"
#include "math/symbolic/symbolic_util.h"

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
  static constexpr size_t kLayerSizeBiased = kLayerSize + 1;
  static constexpr size_t kInputSizeBiased = kInputSize + 1;

  using BiasedSymbolicInputVector =
      Matrix<kInputSizeBiased, 1, symbolic::Expression>;
  using SymbolicOutputVector = Matrix<kOutputSize, 1, symbolic::Expression>;
  using InputVector = Matrix<kInputSize, 1, Number>;
  using OutputVector = Matrix<kOutputSize, 1, Number>;
  using HiddenVector = Matrix<kLayerSize, 1, symbolic::Expression>;
  using BiasedHiddenVector = Matrix<kLayerSizeBiased, 1, symbolic::Expression>;
  using BiasedOutputWeights =
      Matrix<kOutputSize, kLayerSizeBiased, symbolic::Expression>;
  using BiasedHiddenWeights =
      Matrix<kLayerSize, kLayerSizeBiased, symbolic::Expression>;
  using BiasedInputWeights =
      Matrix<kLayerSize, kInputSizeBiased, symbolic::Expression>;

  struct LearningParameters {
    Number learning_rate;
    bool dynamic_learning_rate = false;
  };

  Nnet() {
    std::function<symbolic::Expression(const symbolic::Expression&)>
        activation_function = [](const symbolic::Expression& exp) {
          return symbolic::Sigmoid(exp);
        };

    BiasedSymbolicInputVector inputs = GenInputLayer();

    HiddenVector layer =
        (GenInputLayerWeights() * inputs).Map(activation_function);

    for (size_t i = 1; i <= kNumHiddenLayers; ++i) {
      BiasedHiddenVector biased_layer = AddBias(layer, i);
      layer =
          (GenHiddenLayerWeights(i) * biased_layer).Map(activation_function);
    }

    BiasedHiddenVector biased_layer = AddBias(layer, kNumLayers - 1);
    neural_network_ =
        (GenOutputLayerWeights() * biased_layer).Map(activation_function);

    CalculateInitialWeights();
    CalculateInitialBias();
  }

  static std::string I(size_t i) { return "I(" + std::to_string(i) + ")"; }

  static std::string W(size_t i, size_t j, size_t k) {
    return "W(" + std::to_string(i) + "," + std::to_string(j) + "," +
           std::to_string(k) + ")";
  }

  static std::string B(size_t layer) {
    return "B(" + std::to_string(layer) + ")";
  }

  OutputVector Evaluate(InputVector in) const {
    // Modify weights_ in-place to avoid copying them. This is guaranteed to
    // never have stale inputs (from previous eval) as long as I follow naming
    // conventions and don't fuck up (I probably will).
    symbolic::Environment weights = weights_;
    for (size_t i = 0; i < kInputSize; ++i) {
      weights[I(i)].real() = in.at(i, 0);
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
      weights_[I(i)].real() = in.at(i, 0);
    }

    symbolic::Expression error = GetErrorExpression(o);
    symbolic::Environment weights = weights_;

    Number learning_rate = params.learning_rate;

#pragma omp parallel for shared(error) shared(weights)
    for (size_t layer = 0; layer < kNumLayers; ++layer) {
      for (size_t node = 0; node < LayerSize(layer); ++node) {
        for (size_t edge = 0; edge < PrevLayerSize(layer); ++edge) {
          symbolic::Expression symbolic_gradient =
              error.Derive(W(layer, node, edge));
          symbolic::Expression gradient = symbolic_gradient.Bind(weights);
          auto gradient_value = gradient.Evaluate();
          if (!gradient_value) {
            std::cerr << "Shit" << std::endl;
            for (const std::string& variable : gradient.variables()) {
              std::cerr << variable << std::endl;
            }
          }
          Number weight_update = -gradient_value->real() * learning_rate;
#pragma omp atomic
          weights_[W(layer, node, edge)].real() += weight_update;
        }
      }
    }
  }

  symbolic::Expression GetErrorExpression(OutputVector o) const {
    symbolic::Expression error;
    for (size_t out_idx = 0; out_idx < kOutputSize; ++out_idx) {
      symbolic::Expression output_error =
          neural_network_.at(out_idx, 0) -
          symbolic::Expression(o.at(out_idx, 0));
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
  static constexpr size_t LayerSize(size_t layer_idx) {
    if (layer_idx == kNumHiddenLayers + 1) {
      return kOutputSize;
    }
    return kLayerSize;
  }

  static constexpr size_t BiasedLayerSize(size_t layer_idx) {
    return LayerSize(layer_idx) + 1;
  }

  static constexpr size_t PrevLayerSize(size_t layer_idx) {
    return (layer_idx == 0) ? kInputSize : kLayerSize;
  }

  static constexpr size_t BiasedPrevLayerSize(size_t layer_idx) {
    return PrevLayerSize(layer_idx) + 1;
  }

  template <size_t N>
  Matrix<N + 1, 1, symbolic::Expression> AddBias(
      Matrix<N, 1, symbolic::Expression> x, size_t layer_idx) {
    Matrix<N + 1, 1, symbolic::Expression> biased_layer;
    for (size_t i = 0; i < N; ++i) {
      biased_layer.at(i, 0) = x.at(i, 0);
    }
    biased_layer.at(N, 0) = symbolic::CreateExpression(B(layer_idx));
    return biased_layer;
  }

  void CalculateInitialWeights() {
    for (size_t layer = 0; layer < kNumLayers; ++layer) {
      // Xavier initialization says N(0, 1/kSizeInputs);
      stats::Normal X(0, 1.0 / BiasedPrevLayerSize(layer));
      for (size_t node = 0; node < BiasedLayerSize(layer); ++node) {
        for (size_t edge = 0; edge < BiasedPrevLayerSize(layer); ++edge) {
          weights_[W(layer, node, edge)].real() = X.sample();
        }
      }
    }
  }

  void CalculateInitialBias() {
    for (size_t layer = 0; layer < kNumLayers; ++layer) {
      weights_[B(layer)] = symbolic::NumericValue(1);
    }
  }

  BiasedOutputWeights GenOutputLayerWeights() const {
    BiasedOutputWeights results;
    for (size_t i = 0; i < kOutputSize; ++i) {
      for (size_t j = 0; j < BiasedPrevLayerSize(kNumLayers - 1); ++j) {
        results.at(i, j) =
            symbolic::CreateExpression(W(kNumHiddenLayers + 1, i, j));
      }
    }
    return results;
  }

  BiasedHiddenWeights GenHiddenLayerWeights(const int layer_idx) const {
    BiasedHiddenWeights results;
    for (size_t i = 0; i < LayerSize(layer_idx); ++i) {
      for (size_t j = 0; j < BiasedPrevLayerSize(layer_idx); ++j) {
        results.at(i, j) = symbolic::CreateExpression(W(layer_idx, i, j));
      }
    }
    return results;
  }

  BiasedInputWeights GenInputLayerWeights() const {
    BiasedInputWeights results;
    for (size_t i = 0; i < kLayerSize; ++i) {
      for (size_t j = 0; j < BiasedLayerSize(0); ++j) {
        results.at(i, j) = symbolic::CreateExpression(W(0, i, j));
      }
    }
    return results;
  }

  BiasedSymbolicInputVector GenInputLayer() const {
    BiasedSymbolicInputVector result;
    for (size_t i = 0; i < BiasedLayerSize(i); ++i) {
      result.at(i, 0) = symbolic::CreateExpression(I(i));
    }
    result.at(BiasedLayerSize(0) - 1, 0) = symbolic::CreateExpression(B(0));
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
  symbolic::Environment weights_;  // Also includes biases.
  std::unordered_map<std::string, symbolic::Expression> gradients_;
};

#endif /* NNET_H */

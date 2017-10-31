#ifndef NNET_H
#define NNET_H
#include "math/geometry/matrix.h"
#include "math/symbolic/expression.h"
#include "math/symbolic/symbolic_util.h"

typedef double Number;

// Creates a neural network symbolically. The input layer has no weights. There
// are kNumHiddenLayers + 2 layers with weights (+1 for output layer, +1 for
// first layer which has kInputSize number of weights per neuron).
// TODO(sharf): Implement code generation -- an expression can render itself
// into glsl for parallel execution. That will make this useful.
template <size_t kNumHiddenLayers, size_t kLayerSize, size_t kOutputSize,
          size_t kInputSize>
class Nnet {
  using InputVector = Matrix<kInputSize, 1, symbolic::Expression>;
  using OutputVector = Matrix<kOutputSize, 1, symbolic::Expression>;
  using HiddenVector = Matrix<kOutputSize, 1, symbolic::Expression>;
  using OutputWeights = Matrix<kOutputSize, kLayerSize, symbolic::Expression>;
  using HiddenWeights = Matrix<kLayerSize, kLayerSize, symbolic::Expression>;
  using InputWeights = Matrix<kLayerSize, kInputSize, symbolic::Expression>;

  struct LearningParameters {
    Number learning_rate = 0.1;
  };

  static Nnet XavierInitializedNnet() {}

  static std::string I(size_t i) { return "I(" + std::to_string(i) + ")"; }

  static std::string W(size_t i, size_t j, size_t k) {
    return "W(" + std::to_string(i) + "," + std::to_string(j) + "," +
           std::to_string(k) + ")";
  }

  OutputVector Evaluate(InputVector in) const {
    // This variable countains the bindings for all inputs and weights in the
    // network.
    std::unordered_map<std::string, NumericValue> bindings;

    for (size_t i = 0; i < kInputSize; ++i) {
      bindings[I(i)] = in.at(i, 0);
    }

    bindings.insert(weights_);

    Matrix<kOutputSize, 1, symbolic::Expression> bound_network =
        neural_network_.Bind(bindings);

    auto real_evaluator =
        [](symbolic::Expression exp) {
          auto maybe_value = exp.Evaluate();
          if (!maybe_value) {
            // Shit.
            std::cerr << "Well, fuck, not sure how this happened" << std::endl;
          }
          return maybe_value->real();
        }

    Matrix<kOutputSize, 1, Number>
        results = bound_network.Map(real_evaluator);

    return results;
  }

  // Back propagation
  void Train(InputVector v, OutputVector v, const LearningParameters&) const {}

 private:
  // Zero initialization is bad. Must use factory function with name which is
  // more descriptive of what's going on (e.x., XavierInitializedNnet()).
  Nnet() {
    //// Activation function.
    // auto sigmoid = [](Number x) -> Number {
    //  return 1 / (1 + powf(e, -x))
    //}

    // HiddenVector v = input_weights * in;
    // v = v.Map(sigmoid);
    // for (size_t i = 0; i < kNumLayers; ++i) {
    //  v = hidden_weights[i] * v;
    //  v = v.Map(sigmoid);
    //}
    // OutputVector out = output_weights * v;
    // return out.Map(sigmoid);

    auto activation_function =
        [](symbolic::Expression exp) { return symbolic::Sigmoid(exp); }

    Matrix<kInputSize, 1, symbolic::Expression>
        inputs;

    for (size_t i = 0; i < kInputSize; ++i) {
      inputs.at(i, 0) = symbolic::CreateExpression(I(i));
    }

    Matrix<kLayerSize, kInputSize, symbolic::Expression> input_layer;
    for (size_t i = 0; i < kLayerSize; ++i) {
      for (size_t j = 0; j < kInputSize; ++j) {
        input_layer.at(i, j) = symbolic::CreateExpression(W(0, j));
      }
    }
    input_layer = input_layer.Map(activation_function);

    Matrix<kLayerSize, kLayerSize, symbolic::Expression> layer;
    for (size_t i = 0; i < kLayerSize; ++i) {
      for (size_t j = 0; j < kLayerSize; ++j) {
        hidden_layer.at(i, j) = symbolic::CreateExpression(W(0, i, j));
      }
    }

    for (size_t layer = 0; layer < kNumHiddenLayers; ++layer) {
      Matrix<kLayerSize, kLayerSize, symbolic::Expression> hidden_layer;
      for (size_t i = 0; i < kLayerSize; ++i) {
        for (size_t j = 0; j < kLayerSize; ++j) {
          hidden_layer.at(i, j) = symbolic::CreateExpression(W(i, j, k));
        }
      }
    }
  }

  OutputWeights GenOutputLayerWeights() const {
    OutputWeights results;
    for (size_t i = 0; i < kOutputSize; ++i) {
      for (size_t j = 0; j < kHiddenLayerSize; ++j) {
        // The final layer, which is layer kNumHiddenLayers, is the output
        // layer.
        // There are kNumHiddenLayers + 1 layers, and since they're 0-indexed,
        // this is the final (output) index.
        results.at(i, j) =
            symbolic::CreateExpression(W(kNumHiddenLayers, i, j));
      }
    }
  }

  HiddenWeights GenHiddenLayerWeights(const int layer_idx) const {
    HiddenWeights results;
    for (size_t i = 0; i < kHiddenLayerSize; ++i) {
      for (size_t j = 0; j < kHiddenLayerSize; ++j) {
        results.at(i, j) = symbolic::CreateExpression(W(layer_idx, i, j));
      }
    }
  }

  InputWeights GenInputLayerWeights(const int layer_idx) const {
    InputWeights results;
    for (size_t i = 0; i < kHiddenLayerSize; ++i) {
      for (size_t j = 0; j < kInputSize; ++j) {
        results.at(i, j) = symbolic::CreateExpression(W(layer_idx, i, j));
      }
    }
  }

  InputVector GenInputLayer() const {
    InputVector results;
    for (size_t i = 0; i < kInputSize; ++i) {
      result.at(i, 0) = symbolic::CreateExpression(I(i));
    }
  }

  // The entire neural network is stored symbolically, in a column vector of
  // type symbolic::Expression. To get your outputs, simply call Bind() on all
  // expressions in the column vector and bind weights and inputs.
  //
  // Weight names are of the form:
  // W(i,j,k) -- i is layer number, j is the index of the neuron in that layer.
  // k Is the index of the neuron in the previous layer that's connected to this
  // one (or for the input layer's case, the index of the input).
  // I(i) -- i is the index of the input.
  //
  // All indices and layer numbers are zero-indexed.
  //
  // The first layer has kInputSize nuerons. Every Hidden layer has kLayerSize
  // neurons. The output layer has kOutputSize neurons. There are
  // kNumHiddenLayers + 1 layers with weights (the hidden layers and the output
  // layer).
  Matrix<kOutputSize, 1, symbolic::Expression> neural_network_;
  std::unordered_map<std::string, NumericValue> weights_;
};

#endif /* NNET_H */

#ifndef LAYER_H
#define LAYER_H
#include "math/geometry/matrix.h"
#include "math/stats/normal.h"
#include "math/symbolic/expression.h"
#include "math/symbolic/symbolic_util.h"

#include <array>
#include <cassert>
#include <iterator>
#include <string>
#include <tuple>

namespace nnet {

template <size_t N>
Matrix<N + 1, 1, symbolic::Expression> AddBias(
    Matrix<N, 1, symbolic::Expression> x) {
  Matrix<N + 1, 1, symbolic::Expression> biased_layer;
  for (size_t i = 0; i < N; ++i) {
    biased_layer.at(i, 0) = x.at(i, 0);
  }
  // Bias is always 1.
  biased_layer.at(N, 0) = symbolic::CreateExpression("1");
  return biased_layer;
}

class SymbolGenerator {
 public:
  // TODO(sharf): Come up with good justification for making W non-const.
  virtual std::string W(size_t layer, size_t node, size_t edge) = 0;
  virtual std::string I(size_t i) const = 0;
  virtual std::string O(size_t i) const = 0;
  virtual ~SymbolGenerator() {}
};

// TODO(sharf): Layer also holds weights + state (LayerGenerator -> Layer).
template <size_t kNumInputs, size_t kNumOutputs>
class LayerGenerator {
 public:
  using SymbolicLayer = Matrix<kNumOutputs, 1, symbolic::Expression>;
  using SymbolicInput = Matrix<kNumInputs, 1, symbolic::Expression>;
  using WeightArray = std::array<std::string, kNumOutputs*(kNumInputs + 1)>;

  virtual WeightArray weights() = 0;
  virtual SymbolicLayer GenerateExpression(SymbolicInput input) = 0;
  virtual stats::Normal XavierInitializer() const = 0;
  virtual ~LayerGenerator() {}

 protected:
  LayerGenerator(SymbolGenerator* generator, size_t layer_index)
      : generator_(generator), layer_index_(layer_index) {}

  SymbolGenerator* generator_;
  size_t layer_index_;
};

template <size_t kNumInputs, size_t kNumOutputs>
class FeedForwardLayer : public LayerGenerator<kNumInputs, kNumOutputs> {
 public:
  // Reference objects in superclass with Super::
  using Super = LayerGenerator<kNumInputs, kNumOutputs>;
  using WeightArray = typename Super::WeightArray;
  using SymbolicLayer = typename Super::SymbolicLayer;
  using SymbolicInput = typename Super::SymbolicInput;

  // +1 for bias.
  using SymbolicWeights =
      Matrix<kNumOutputs, kNumInputs + 1, symbolic::Expression>;

  using ActivationFunctionType =
      std::function<symbolic::Expression(const symbolic::Expression&)>;

  FeedForwardLayer(const ActivationFunctionType& activation_function,
                   SymbolGenerator* generator, size_t layer_index)
      : Super(generator, layer_index),
        activation_function_(activation_function) {}

  FeedForwardLayer(SymbolGenerator* generator, size_t layer_index)
      : Super(generator, layer_index) {
    activation_function_ = [](const symbolic::Expression& exp) {
      return symbolic::Sigmoid(exp);
    };
  }

  WeightArray weights() override {
    WeightArray weights;
    size_t back_index = 0;
    for (size_t i = 0; i < kNumOutputs; ++i) {
      // + 1 for Bias.
      for (size_t j = 0; j < kNumInputs + 1; ++j) {
        assert(back_index < weights.size());
        weights[back_index++] = Super::generator_->W(Super::layer_index_, i, j);
      }
    }
    return weights;
  }

  SymbolicLayer GenerateExpression(SymbolicInput input) override {
    SymbolicWeights weight_matrix;
    for (size_t i = 0; i < kNumOutputs; ++i) {
      for (size_t j = 0; j < kNumInputs; ++j) {
        weight_matrix.at(i, j) = symbolic::CreateExpression(
            Super::generator_->W(Super::layer_index_, i, j));
      }
    }

    // Add layer bias.
    auto biased_input = AddBias(input);
    for (size_t i = 0; i < kNumOutputs; ++i) {
      // Bias is the final column in the SymbolicWeights matrix. Since size
      // is kNumInputs + 1 and it is zero-indexed, kNumInputs is the final
      // index (index of bias).
      weight_matrix.at(i, kNumInputs) = symbolic::CreateExpression(
          Super::generator_->W(Super::layer_index_, i, kNumInputs));
    }

    return (weight_matrix * biased_input).Map(activation_function_);
  }

  stats::Normal XavierInitializer() const override {
    // + 1 for implicit bias input.
    return stats::Normal(0, 1.0 / (kNumInputs + 1));
  }

  // This function will be used to map the activation function to a matrix
  // of symbolic expressions.
  std::function<symbolic::Expression(const symbolic::Expression&)>
      activation_function_;
};

}  // namespace nnet
#endif /* LAYER_H */

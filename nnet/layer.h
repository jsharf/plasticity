#ifndef LAYER_H
#define LAYER_H
#include "math/geometry/dynamic_matrix.h"
#include "math/stats/normal.h"
#include "math/symbolic/expression.h"
#include "math/symbolic/symbolic_util.h"

#include <array>
#include <cassert>
#include <iterator>
#include <string>
#include <tuple>
#include <vector>

namespace nnet {

Matrix<symbolic::Expression> AddBias(Matrix<symbolic::Expression> x) {
  auto dim = x.size();
  size_t rows = std::get<0>(dim);
  size_t cols = std::get<1>(dim);
  if (cols != 1) {
    std::cerr << "Err: AddBias must only be called on column vectors!"
              << std::endl;
    std::exit(1);
  }
  Matrix<symbolic::Expression> biased_layer(std::get<0>(dim) + 1, 1);
  for (size_t i = 0; i < rows; ++i) {
    biased_layer.at(i, 0) = x.at(i, 0);
  }
  // Bias is always 1.
  biased_layer.at(rows, 0) = symbolic::CreateExpression("1");
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
class LayerGenerator {
 public:
  struct Dimensions {
    size_t num_inputs;
    size_t num_outputs;
  };

  // Dim(num_outputs * (num_inputs + 1))
  using WeightArray = std::vector<std::string>;

  virtual WeightArray weights() = 0;
  virtual Matrix<symbolic::Expression> GenerateExpression(
      Matrix<symbolic::Expression> input) = 0;
  virtual stats::Normal XavierInitializer() const = 0;
  Dimensions GetDimensions() const { return dimensions_; }
  virtual ~LayerGenerator() {}

 protected:
  LayerGenerator(const Dimensions& dimensions, SymbolGenerator* generator,
                 size_t layer_index)
      : dimensions_(dimensions),
        generator_(generator),
        layer_index_(layer_index) {}

  Dimensions dimensions_;
  SymbolGenerator* generator_;
  size_t layer_index_;
};

class FeedForwardLayer : public LayerGenerator {
 public:
  // Reference objects in superclass with Super::
  using Super = LayerGenerator;
  using WeightArray = typename Super::WeightArray;
  using Dimensions = typename Super::Dimensions;

  using ActivationFunctionType =
      std::function<symbolic::Expression(const symbolic::Expression&)>;

  FeedForwardLayer(const Dimensions& dimensions,
                   const ActivationFunctionType& activation_function,
                   SymbolGenerator* generator, size_t layer_index)
      : Super(dimensions, generator, layer_index),
        activation_function_(activation_function) {}

  FeedForwardLayer(const Dimensions& dimensions, SymbolGenerator* generator,
                   size_t layer_index)
      : Super(dimensions, generator, layer_index) {
    activation_function_ = [](const symbolic::Expression& exp) {
      return symbolic::Sigmoid(exp);
    };
  }

  WeightArray weights() override {
    WeightArray weights(dimensions_.num_outputs * (dimensions_.num_inputs + 1));
    size_t back_index = 0;
    for (size_t i = 0; i < dimensions_.num_outputs; ++i) {
      // + 1 for Bias.
      for (size_t j = 0; j < dimensions_.num_inputs + 1; ++j) {
        assert(back_index < weights.size());
        weights[back_index++] = Super::generator_->W(Super::layer_index_, i, j);
      }
    }
    return weights;
  }

  Matrix<symbolic::Expression> GenerateExpression(Matrix<symbolic::Expression> input) override {
    auto dim = input.size();
    size_t rows = std::get<0>(dim);
    size_t cols = std::get<1>(dim);
    if ((rows != dimensions_.num_inputs) || (cols != 1)) {
      std::cerr << "Error: LayerGenerator::GenerateExpression called on input "
                   "of incorrect size: "
                << "(" << rows << ", " << cols << ")" << std::endl;
      std::exit(1);
    }
    // +1 in number of columns to account for added bias weights.
    Matrix<symbolic::Expression> weight_matrix(dimensions_.num_outputs,
                                               dimensions_.num_inputs + 1);
    for (size_t i = 0; i < dimensions_.num_outputs; ++i) {
      for (size_t j = 0; j < dimensions_.num_inputs; ++j) {
        weight_matrix.at(i, j) = symbolic::CreateExpression(
            Super::generator_->W(Super::layer_index_, i, j));
      }
    }

    // Add layer bias.
    auto biased_input = AddBias(input);
    for (size_t i = 0; i < dimensions_.num_outputs; ++i) {
      // Bias is the final column in the weights matrix. Since size
      // is kNumInputs + 1 and it is zero-indexed, kNumInputs is the final
      // index (index of bias).
      weight_matrix.at(i, dimensions_.num_inputs) = symbolic::CreateExpression(
          Super::generator_->W(Super::layer_index_, i, dimensions_.num_inputs));
    }

    return (weight_matrix * biased_input).Map(activation_function_);
  }

  stats::Normal XavierInitializer() const override {
    // + 1 for implicit bias input.
    return stats::Normal(0, 1.0 / (dimensions_.num_inputs + 1));
  }

  // This function will be used to map the activation function to a matrix
  // of symbolic expressions.
  std::function<symbolic::Expression(const symbolic::Expression&)>
      activation_function_;
};

}  // namespace nnet
#endif /* LAYER_H */

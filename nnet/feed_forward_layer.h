#ifndef FEED_FORWARD_LAYER_H
#define FEED_FORWARD_LAYER_H
#include "math/geometry/dynamic_matrix.h"
#include "math/nnet/layer_impl.h"
#include "math/stats/normal.h"
#include "math/symbolic/expression.h"
#include "math/symbolic/symbolic_util.h"

namespace nnet {

class FeedForwardLayer : public LayerImpl {
 public:
  // Reference objects in superclass with Super::
  using Super = LayerImpl;
  using WeightArray = typename Super::WeightArray;
  using Dimensions = typename Super::Dimensions;

  using ActivationFunctionType = LayerImpl::ActivationFunctionType;

  FeedForwardLayer(const Dimensions& dimensions,
                   const ActivationFunctionType& activation_function,
                   SymbolGenerator* generator, size_t layer_index);

  FeedForwardLayer(const Dimensions& dimensions, SymbolGenerator* generator,
                   size_t layer_index)
      : Super(dimensions, generator, layer_index) {
    activation_function_ = [](const symbolic::Expression& exp) {
      return symbolic::Sigmoid(exp);
    };
  }

  WeightArray weights() const override;

  Matrix<symbolic::Expression> GenerateExpression(
      const Matrix<symbolic::Expression>& input) override;

  stats::Normal XavierInitializer() const override {
    // + 1 for implicit bias input.
    return stats::Normal(0, 1.0 / (dimensions_.num_inputs + 1));
  }

  std::unique_ptr<LayerImpl> Clone() const override;

  // This function will be used to map the activation function to a matrix
  // of symbolic expressions.
  std::function<symbolic::Expression(const symbolic::Expression&)>
      activation_function_;
};

}  // namespace nnet

#endif /* FEED_FORWARD_LAYER_H */

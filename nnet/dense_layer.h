#ifndef DENSE_LAYER_H
#define DENSE_LAYER_H
#include "math/geometry/dynamic_matrix.h"
#include "math/nnet/layer_dimensions.h"
#include "math/nnet/layer_impl.h"
#include "math/nnet/symbol_generator.h"
#include "math/stats/normal.h"
#include "math/symbolic/expression.h"
#include "math/symbolic/symbolic_util.h"

namespace nnet {

class DenseLayer : public LayerImpl {
 public:
  // Reference objects in superclass with Super::
  using Super = LayerImpl;
  using WeightArray = typename Super::WeightArray;

  using ActivationFunctionType = LayerImpl::ActivationFunctionType;

  DenseLayer(const Dimensions& dimensions,
                   const ActivationFunctionType& activation_function,
                   size_t layer_index);

  DenseLayer(const Dimensions& dimensions, size_t layer_index)
      : Super(dimensions, layer_index),
        generator_(dimensions),
        activation_function_(symbolic::Sigmoid) {}

  WeightArray weights() const override;

  Matrix<symbolic::Expression> GenerateExpression(
      const Matrix<symbolic::Expression>& input) override;

  std::unique_ptr<LayerImpl> Clone() const override;

 private:
  DenseSymbolGenerator generator_;

  // This function will be used to map the activation function to a matrix
  // of symbolic expressions.
  std::function<symbolic::Expression(const symbolic::Expression&)>
      activation_function_;
};

}  // namespace nnet

#endif /* DENSE_LAYER_H */

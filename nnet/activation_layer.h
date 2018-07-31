#ifndef ACTIVATION_LAYER_H
#define ACTIVATION_LAYER_H
#include "math/geometry/dynamic_matrix.h"
#include "math/nnet/layer_dimensions.h"
#include "math/nnet/layer_impl.h"
#include "math/stats/normal.h"
#include "math/symbolic/expression.h"
#include "math/symbolic/symbolic_util.h"

namespace nnet {

class ActivationLayer : public LayerImpl {
 public:
  using Super = LayerImpl;
  using ActivationFunctionType = typename Super::ActivationFunctionType;

  ActivationLayer(size_t size,
                  const ActivationFunctionType& activation_function,
                  size_t layer_index)
      : Super(Dimensions{size, size}, layer_index),
        activation_function_(activation_function) {}

  Matrix<symbolic::Expression> GenerateExpression(
      const Matrix<symbolic::Expression>& input) const override;

  std::unique_ptr<LayerImpl> Clone() const override;

 private:
  ActivationFunctionType activation_function_;
};

}  // namespace nnet

#endif /* ACTIVATION_LAYER_H */

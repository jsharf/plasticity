#ifndef SOFTMAX_LAYER_H
#define SOFTMAX_LAYER_H

#include "math/geometry/dynamic_matrix.h"
#include "math/nnet/layer_dimensions.h"
#include "math/nnet/layer_impl.h"
#include "math/stats/normal.h"
#include "math/symbolic/expression.h"
#include "math/symbolic/symbolic_util.h"

namespace nnet {

class SoftmaxLayer : public LayerImpl {
 public:
  using Super = LayerImpl;
  using WeightArray = typename Super::WeightArray;

  SoftmaxLayer(size_t size, size_t layer_index)
      : Super(Dimensions{size, size}, layer_index) {}

  WeightArray weights() const override { return {}; }

  Matrix<symbolic::Expression> GenerateExpression(
      const Matrix<symbolic::Expression>& input) override;

  std::unique_ptr<LayerImpl> Clone() const override;
};

}  // namespace nnet

#endif /* SOFTMAX_LAYER_H */

#ifndef ACTIVATION_LAYER_H
#define ACTIVATION_LAYER_H
#include "math/geometry/dynamic_matrix.h"
#include "math/nnet/layer_impl.h"
#include "math/stats/normal.h"
#include "math/symbolic/expression.h"
#include "math/symbolic/symbolic_util.h"

namespace nnet {

class ActivationLayer : public LayerImpl {
 public:
  using Super = LayerImpl;
  using WeightArray = typename Super::WeightArray;
  using Dimensions = typename Super::Dimensions;
  using ActivationFunctionType = typename Super::ActivationFunctionType;

  ActivationLayer(size_t size,
                  const ActivationFunctionType& activation_function,
                  SymbolGenerator* generator, size_t layer_index)
      : Super(Dimensions{size, size}, generator, layer_index),
        activation_function_(activation_function) {}

  WeightArray weights() override { return {}; }

  Matrix<symbolic::Expression> GenerateExpression(
      const Matrix<symbolic::Expression>& input) override;

  stats::Normal XavierInitializer() const override {
    // No weights in an activation layer.
    std::cerr << "Warning: XavierInitializer() called on Activation layer"
              << std::endl;
    return stats::Normal(0, 0);
  }

  std::unique_ptr<LayerImpl> Clone() const override;

 private:
  ActivationFunctionType activation_function_;
};

}  // namespace nnet

#endif /* ACTIVATION_LAYER_H */

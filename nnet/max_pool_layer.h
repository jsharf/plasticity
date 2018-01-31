#ifndef MAX_POOL_LAYER_H
#define MAX_POOL_LAYER_H
#include "math/geometry/dynamic_matrix.h"
#include "math/nnet/convolution_layer.h"
#include "math/nnet/layer_impl.h"
#include "math/stats/normal.h"
#include "math/symbolic/expression.h"
#include "math/symbolic/symbolic_util.h"

namespace nnet {

class MaxPoolLayer : public LayerImpl {
 public:
  using Super = LayerImpl;
  using WeightArray = typename Super::WeightArray;
  using LinearDimensions = typename Super::Dimensions;
  using FilterParams = ConvolutionLayer::Dimensions;
  using VolumeDimensions = ConvolutionLayer::VolumeDimensions;

  struct AreaDimensions {
    size_t width, height;
  };

  static LinearDimensions GenLinearDimensions(const VolumeDimensions& dim,
                                              const AreaDimensions& output) {
    return LinearDimensions{
        // Num Inputs.
        dim.width * dim.height * dim.depth,
        // Num Outputs.
        output.width * output.height * dim.depth,
    };
  }

  // Returns output volume dim (width, height, depth).
  static std::tuple<size_t, size_t, size_t> GetOutputDimensions(
      const VolumeDimensions& dim, const AreaDimensions& output);

  MaxPoolLayer(const VolumeDimensions& input, const AreaDimensions& output,
               SymbolGenerator* generator, size_t layer_index);

  WeightArray weights() override { return {}; }

  Matrix<symbolic::Expression> GenerateExpression(
      const Matrix<symbolic::Expression>& input) override;

  stats::Normal XavierInitializer() const override {
    // No weights in a maxpool layer.
    std::cerr << "Warning: XavierInitializer() called on MaxPool layer"
              << std::endl;
    return stats::Normal(0, 0);
  }

  std::unique_ptr<LayerImpl> Clone() const override;

 private:
  VolumeDimensions input_;
  AreaDimensions target_;
};

}  // namespace nnet

#endif /* MAX_POOL_LAYER_H */

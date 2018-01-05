#ifndef LAYER_H
#define LAYER_H
#include "math/geometry/dynamic_matrix.h"
#include "math/nnet/convolution_layer.h"
#include "math/nnet/feed_forward_layer.h"
#include "math/nnet/layer_impl.h"
#include "math/stats/normal.h"
#include "math/symbolic/expression.h"
#include "math/symbolic/symbolic_util.h"

#include <array>
#include <cassert>
#include <iterator>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

namespace nnet {

// Holds a pointer to a Layer and manages the resources.
class Layer {
 public:
  // Public exported types
  using WeightArray = LayerImpl::WeightArray;
  using Dimensions = LayerImpl::Dimensions;
  using ActivationFunctionType = LayerImpl::ActivationFunctionType;
  using VolumeDimensions = ConvolutionLayer::VolumeDimensions;
  using FilterParams = ConvolutionLayer::FilterParams;

  // Constructors
  Layer() {}
  Layer(std::unique_ptr<LayerImpl>&& root);
  explicit Layer(Layer&& rhs);

  // FeedForward Layer constructors.
  static Layer FeedForwardLayer(
      size_t layer_index, const Dimensions& dimensions,
      const ActivationFunctionType& activation_function,
      SymbolGenerator* generator);
  static Layer FeedForwardLayer(size_t layer_index,
                                const Dimensions& dimensions,
                                SymbolGenerator* generator);

  // Convolutional Layer constructors.
  static Layer ConvolutionLayer(size_t layer_index,
                                const VolumeDimensions& dimensions,
                                const FilterParams& params,
                                SymbolGenerator* generator);

  WeightArray weights();
  Matrix<symbolic::Expression> GenerateExpression(
      const Matrix<symbolic::Expression>& input);
  stats::Normal XavierInitializer();
  Dimensions GetDimensions() const { return impl_->GetDimensions(); }

 private:
  std::unique_ptr<LayerImpl> impl_;
};

}  // namespace nnet
#endif /* LAYER_H */

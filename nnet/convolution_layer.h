#ifndef CONVOLUTION_LAYER_H
#define CONVOLUTION_LAYER_H
#include "math/geometry/dynamic_matrix.h"
#include "math/nnet/layer_impl.h"
#include "math/nnet/symbol_generator.h"
#include "math/nnet/layer_dimensions.h"
#include "math/stats/normal.h"
#include "math/symbolic/expression.h"
#include "math/symbolic/symbolic_util.h"

#include <tuple>

namespace nnet {

// Parameter sharing is enforced. Each filter has its own weights, but the
// weights are the same regardless of which part of the input is convolved with
// the filter.
class ConvolutionLayer : public LayerImpl {
 public:
  // Reference objects in superclass with Super::
  using Super = LayerImpl;
  // (x, y, {r, g, b}) -> index.
  using IndexMap = std::function<size_t(size_t, size_t, size_t)>;

  static LinearDimensions GenLinearDimensions(const VolumeDimensions& dim,
                                              const FilterParams& filters);

  // Returns output volume dim (width, height, depth).
  static std::tuple<size_t, size_t, size_t> GetOutputDimensions(
      const VolumeDimensions& dim, const FilterParams& filters);

  // TODO(sharf): Specifying IndexMaps really sucks. Figure out a better
  // interface for specifying the format of the image.
  ConvolutionLayer(const VolumeDimensions& dimensions,
                   const FilterParams& filters, IndexMap input_map,
                   IndexMap output_map, size_t layer_index);

  ConvolutionLayer(const VolumeDimensions& dimensions,
                   const FilterParams& filters, size_t layer_index);

  const std::vector<std::string>& weights() const override;

  Matrix<symbolic::Expression> GenerateExpression(
      const Matrix<symbolic::Expression>& input) const override;

  std::unique_ptr<LayerImpl> Clone() const override;

 private:
  ConvSymbolGenerator generator_;
  FilterParams filters_;
  VolumeDimensions imdim_;
  IndexMap input_map_;
  IndexMap output_map_;
};

}  // namespace nnet

#endif  // CONVOLUTION_LAYER_H

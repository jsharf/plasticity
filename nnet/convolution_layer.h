#ifndef CONVOLUTION_LAYER_H
#define CONVOLUTION_LAYER_H
#include "math/geometry/dynamic_matrix.h"
#include "math/nnet/layer_impl.h"
#include "math/nnet/symbol_generator.h"
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
  using WeightArray = typename Super::WeightArray;
  using LinearDimensions = typename Super::Dimensions;
  using IndexMap = std::function<size_t(size_t, size_t, size_t)>;

  struct FilterParams {
    // Dimensions of each filter.
    size_t width;
    size_t height;
    size_t depth;

    // Filter stride. PS If not sure, set to 1.
    size_t stride;

    // Zero-padding on input image. This is the number of zeroes (pixels) added
    // to *each* side of the input image when doing a convolution with the
    // filter.
    size_t padding;

    // The number of filters.
    size_t num_filters;
  };

  struct VolumeDimensions {
    size_t width;
    size_t height;

    // 1 for grey, 3 for rgb. Or whatever, it's really just an input volume,
    // this is a convolutional layer.
    size_t depth;
  };

  static LinearDimensions GenLinearDimensions(const VolumeDimensions& dim,
                                              const FilterParams& filters);

  // Returns output volume dim (width, height, depth).
  static std::tuple<size_t, size_t, size_t> GetOutputDimensions(
      const VolumeDimensions& dim, const FilterParams& filters);

  // TODO(sharf): Specifying IndexMaps really sucks. Figure out a better
  // interface for specifying the format of the image.
  ConvolutionLayer(const VolumeDimensions& dimensions,
                   const FilterParams& filters, IndexMap input_map,
                   IndexMap output_map, SymbolGenerator* generator,
                   size_t layer_index);

  ConvolutionLayer(const VolumeDimensions& dimensions,
                   const FilterParams& filters, SymbolGenerator* generator,
                   size_t layer_index);

  WeightArray weights() const override;

  Matrix<symbolic::Expression> GenerateExpression(
      const Matrix<symbolic::Expression>& input) override;

  stats::Normal XavierInitializer() const override;

  std::unique_ptr<LayerImpl> Clone() const override;

 private:
  FilterParams filters_;
  VolumeDimensions imdim_;
  IndexMap input_map_;
  IndexMap output_map_;
};

}  // namespace nnet

#endif  // CONVOLUTION_LAYER_H

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

  static LinearDimensions GenLinearDimensions(const VolumeDimensions& dim,
                                              const FilterParams& filters);

  // Returns output volume dim (width, height, depth).
  static std::tuple<size_t, size_t, size_t> GetOutputDimensions(
      const VolumeDimensions& dim, const FilterParams& filters);

  ConvolutionLayer(const VolumeDimensions& dimensions,
                   const FilterParams& filters, size_t layer_index);

  const std::vector<std::string>& weights() const override;

  symbolic::Expression GenerateOutputCode(
      const symbolic::Expression& index) const override;

  symbolic::Expression GenerateGradientCode(const symbolic::Expression& index) const;

  Matrix<symbolic::Expression> InputGradientsForOutput(const symbolic::Expression& index) const;

  Matrix<symbolic::Expression> WeightGradientsForOutput(const symbolic::Expression& index) const;

  std::unique_ptr<LayerImpl> Clone() const override;

 private:
  ConvSymbolGenerator generator_;
  FilterParams filters_;
  VolumeDimensions imdim_;
};

}  // namespace nnet

#endif  // CONVOLUTION_LAYER_H

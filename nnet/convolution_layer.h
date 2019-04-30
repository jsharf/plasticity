#ifndef CONVOLUTION_LAYER_H
#define CONVOLUTION_LAYER_H
#include "math/codegen/codegen.h"
#include "math/geometry/dynamic_matrix.h"
#include "math/nnet/layer_impl.h"
#include "math/nnet/symbol_generator.h"
#include "math/nnet/layer_dimensions.h"
#include "math/stats/normal.h"
#include "math/symbolic/expression.h"
#include "math/symbolic/symbolic_util.h"

#include <tuple>

namespace nnet {

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

  void GenerateOutputCode(
      const symbolic::Expression& index, codegen::Generator *cg) const override;
  void WeightGradientCode(const symbolic::Expression &index,
                          codegen::Generator *cg) const override;
  void InputGradientCode(const symbolic::Expression &index,
                         codegen::Generator *cg) const override;

  std::unique_ptr<LayerImpl> Clone() const override;

  std::string layer_type() const override {
    return "convolution_layer";
  }

 private:
   std::tuple<symbolic::Expression, symbolic::Expression>
   GetOutputCoordinates(const symbolic::Expression &input_row,
                        const symbolic::Expression &input_col) const;
   std::tuple<symbolic::Expression, symbolic::Expression>
   GetInputCoordinates(const symbolic::Expression &output_row,
                       const symbolic::Expression &output_col) const;
   ConvSymbolGenerator generator_;
   FilterParams filters_;
   VolumeDimensions imdim_;
};

}  // namespace nnet

#endif  // CONVOLUTION_LAYER_H

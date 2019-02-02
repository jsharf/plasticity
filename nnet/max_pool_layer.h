#ifndef MAX_POOL_LAYER_H
#define MAX_POOL_LAYER_H
#include "math/codegen/codegen.h"
#include "math/geometry/dynamic_matrix.h"
#include "math/nnet/convolution_layer.h"
#include "math/nnet/layer_dimensions.h"
#include "math/nnet/layer_impl.h"
#include "math/nnet/symbol_generator.h"
#include "math/stats/normal.h"
#include "math/symbolic/expression.h"
#include "math/symbolic/symbolic_util.h"

namespace nnet {

class MaxPoolLayer : public LayerImpl {
 public:
  using Super = LayerImpl;

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
  static std::tuple<size_t, size_t, size_t>
  GetOutputDimensions(const VolumeDimensions &dim,
                      const AreaDimensions &output);

  MaxPoolLayer(const VolumeDimensions &input, const AreaDimensions &output,
               size_t layer_index);

  symbolic::Expression
  GenerateOutputSymbol(const symbolic::Expression &index) const;

  void GenerateOutputCode(const symbolic::Expression &index,
                          codegen::Generator *cg) const override;

  void InputGradientCode(const symbolic::Expression &input_index,
                         codegen::Generator *cg) const override;

  void WeightGradientCode(const symbolic::Expression &weight_index,
                          codegen::Generator *cg) const override;

  std::unique_ptr<LayerImpl> Clone() const override;

  std::string layer_type() const override {
    return "max_pool_layer";
  }

 private:
  InputVolumeSymbolGenerator generator_;
  VolumeDimensions input_;
  AreaDimensions target_;
};

}  // namespace nnet

#endif /* MAX_POOL_LAYER_H */

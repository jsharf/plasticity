#ifndef DENSE_LAYER_H
#define DENSE_LAYER_H
#include "math/codegen/codegen.h"
#include "math/geometry/dynamic_matrix.h"
#include "math/nnet/layer_dimensions.h"
#include "math/nnet/layer_impl.h"
#include "math/nnet/symbol_generator.h"
#include "math/stats/normal.h"
#include "math/symbolic/expression.h"
#include "math/symbolic/symbolic_util.h"

namespace nnet {

// NOTE: This class alone is a linear net sum. In other words, it's a dense
// layer *without* an activation function. Architecture.h will pair DenseLayer
// with activation functions when building models, however they're actually
// separate layers behind the seems. This simplifies the backprop by adding
// another step (application of the chain rule).
class DenseLayer : public LayerImpl {
 public:
  // Reference objects in superclass with Super::
  using Super = LayerImpl;

  DenseLayer(const Dimensions& dimensions, size_t layer_index)
      : Super(dimensions, layer_index),
        generator_(dimensions),
        dimensions_(dimensions) {}

  const std::vector<std::string>& weights() const override;

  void GenerateOutputCode(const symbolic::Expression &index,
                          codegen::Generator* cg) const override;

  void InputGradientCode(const symbolic::Expression &input_index,
                         codegen::Generator* cg) const override;

  void WeightGradientCode(const symbolic::Expression &weight_index,
                          codegen::Generator* cg) const override;

  std::unique_ptr<LayerImpl> Clone() const override;

  std::string layer_type() const override {
    return "dense_layer";
  }

 private:
  DenseSymbolGenerator generator_;
  Dimensions dimensions_;

  symbolic::Expression OutputSymbol(const symbolic::Expression &output_index) const;
};

}  // namespace nnet

#endif /* DENSE_LAYER_H */

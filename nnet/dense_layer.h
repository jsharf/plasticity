#ifndef DENSE_LAYER_H
#define DENSE_LAYER_H
#include "codegen/codegen.h"
#include "geometry/dynamic_matrix.h"
#include "nnet/layer_dimensions.h"
#include "nnet/layer_impl.h"
#include "nnet/symbol_generator.h"
#include "stats/normal.h"
#include "symbolic/expression.h"
#include "symbolic/symbolic_util.h"

namespace nnet {

// NOTE: This class alone is a dense layer *without* an activation function.
// the Architecture class defined in architecture.h will pair DenseLayer with
// activation functions when building models, however they're actually separate
// layers behind the scene. This simplifies implementation a bit.
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
};

}  // namespace nnet

#endif /* DENSE_LAYER_H */

#ifndef SOFTMAX_LAYER_H
#define SOFTMAX_LAYER_H
#include "plasticity/codegen/codegen.h"
#include "plasticity/geometry/dynamic_matrix.h"
#include "plasticity/nnet/layer_dimensions.h"
#include "plasticity/nnet/layer_impl.h"
#include "plasticity/nnet/symbol_generator.h"
#include "plasticity/stats/normal.h"
#include "plasticity/symbolic/expression.h"
#include "plasticity/symbolic/symbolic_util.h"

namespace nnet {

class SoftmaxLayer : public LayerImpl {
 public:
  using Super = LayerImpl;

  SoftmaxLayer(size_t size, size_t layer_index)
      : Super(Dimensions{size, size}, layer_index) {}

  void GenerateOutputCode(const symbolic::Expression &index,
                          codegen::Generator *cg) const;

  void InputGradientCode(const symbolic::Expression &input_index,
                         codegen::Generator *cg) const override;

  void WeightGradientCode(const symbolic::Expression &weight_index,
                          codegen::Generator *cg) const override;

  std::unique_ptr<LayerImpl> Clone() const override;

  std::string layer_type() const override {
    return "softmax_layer";
  }

 private:
  symbolic::Expression GenerateOutputSymbol(
      const symbolic::Expression &index, const symbolic::Expression &max) const;

  symbolic::Expression AppendMaxCode(codegen::Generator* cg) const;

  SymbolGenerator generator_;
};

}  // namespace nnet

#endif /* SOFTMAX_LAYER_H */

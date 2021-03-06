#ifndef ACTIVATION_LAYER_H
#define ACTIVATION_LAYER_H
#include "codegen/codegen.h"
#include "geometry/dynamic_matrix.h"
#include "nnet/layer_dimensions.h"
#include "nnet/layer_impl.h"
#include "nnet/symbol_generator.h"
#include "stats/normal.h"
#include "symbolic/expression.h"
#include "symbolic/symbolic_util.h"

namespace nnet {

class ActivationLayer : public LayerImpl {
 public:
  using Super = LayerImpl;
  using ActivationFunctionType = typename Super::ActivationFunctionType;

  ActivationLayer(size_t size,
                  const ActivationFunctionType& activation_function,
                  size_t layer_index)
      : Super(Dimensions{size, size}, layer_index),
        activation_function_(activation_function) {}

  void GenerateOutputCode(const symbolic::Expression &index,
                          codegen::Generator *cg) const override;

  void InputGradientCode(const symbolic::Expression &input_index,
                         codegen::Generator *cg) const override;

  void WeightGradientCode(const symbolic::Expression &weight_index,
                          codegen::Generator *cg) const override;

  std::unique_ptr<LayerImpl> Clone() const override;

  std::string layer_type() const override {
    return "activation_layer";
  }

 private:
  ActivationFunctionType activation_function_;
  SymbolGenerator generator_;
};

}  // namespace nnet

#endif /* ACTIVATION_LAYER_H */

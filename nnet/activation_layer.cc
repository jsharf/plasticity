#include <memory>
#include "math/nnet/activation_layer.h"

namespace nnet {

void ActivationLayer::GenerateOutputCode(
    const symbolic::Expression& index) const {
  symbolic::Expression retval = activation_function_(generator_.I(index));

  codegen::CudaGenerator cg;
  cg.AppendLineOfCode("return " + retval.to_string() + cg.linesep());
  return cg;
}

void ActivationLayer::InputGradientCode(const symbolic::Expression &input_index,
                                        codegen::Generator *cg) const {
  symbolic::Expression output =
      activation_function_(generator_.I(input_index));
  symbolic::Expression deriv =
      output.Derive(generator_.I(input_index).to_string());
  symbolic::Expression retval = generator_.GRADIENT(input_index) * deriv;

  cg->AppendLineOfCode("return " + retval.to_string() + cg->linesep());
}

void ActivationLayer::WeightGradientCode(
    const symbolic::Expression &weight_index, codegen::Generator *cg) const {
  cg->AppendLineOfCode("return 0.0" + cg->linesep());
}

std::unique_ptr<LayerImpl> ActivationLayer::Clone() const {
  return std::make_unique<ActivationLayer>(dimensions_.num_inputs,
                                           activation_function_, layer_index_);
}

}  // namespace nnet

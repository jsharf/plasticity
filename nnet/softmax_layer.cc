#include "math/nnet/softmax_layer.h"
#include <memory>

namespace nnet {

void SoftmaxLayer::GenerateOutputCode(const symbolic::Expression &index,
                                      codegen::Generator *cg) const {
  symbolic::Expression expsum = symbolic::NumericValue(0.0); 

  for (size_t i = 0; i < dimensions_.num_inputs; ++i) {
    expsum = expsum + symbolic::Exp(Expression::CreateNumericValue(generator_.I(i)));
  }

  symbolic::Expression retval =
      symbolic::Exp(symbolic::Expression::CreateNumericValue(
          "I[" + index.to_string() + "]")) /
      expsum;

  cg->AppendLineOfCode("return " + retval.to_string() + cg.linesep());
}

void SoftmaxLayer::InputGradientCode(const symbolic::Expression &input_index,
                                     codegen::Generator *cg) const {
  symbolic::Expression output = GenerateOutputCode(input_index);
  symbolic::Expression deriv =
      output.Derive(generator_.I(input_index).to_string());
  symbolic::Expression retval = generator_.GRADIENT(input_index) * deriv;

  cg->AppendLineOfCode("return " + retval.to_string() + cg->linesep());
}

void SoftmaxLayer::WeightGradientCode(const symbolic::Expression &weight_index,
                                      codegen::Generator *cg) const {
  cg->AppendLineOfCode("Return 0.0" + cg->linesep());
}

std::unique_ptr<LayerImpl> SoftmaxLayer::Clone() const {
  return std::make_unique<SoftmaxLayer>(dimensions_.num_inputs, layer_index_);
}

}  // namespace nnet

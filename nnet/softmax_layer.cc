#include "nnet/softmax_layer.h"
#include <memory>

namespace nnet {

// Max is subtracted from numerator and denominator in order to make softmax
// numerically stable (No positive values, so no worry of exp(x) overflowing.
// Also, one zero value means that there's no worry of vanishing denominator)
symbolic::Expression SoftmaxLayer::GenerateOutputSymbol(
    const symbolic::Expression &index, const symbolic::Expression &max) const {
  symbolic::Expression expsum = symbolic::NumericValue(0.0); 

  for (size_t i = 0; i < dimensions_.num_inputs; ++i) {
    expsum =
        expsum + symbolic::Exp(Expression::CreateNumericValue(generator_.I(i)) - max);
  }

  return symbolic::Exp(symbolic::Expression::CreateNumericValue(
             "I[" + index.to_string() + "]") - max) /
         expsum;
}

void SoftmaxLayer::GenerateOutputCode(const symbolic::Expression &index,
                                      codegen::Generator *cg) const {

  symbolic::Expression max = AppendMaxCode(cg);
  symbolic::Expression retval = GenerateOutputSymbol(index, max);

  cg->AppendLineOfCode("return " + retval.to_string() + cg->linesep());
}

symbolic::Expression SoftmaxLayer::AppendMaxCode(codegen::Generator* cg) const {
  symbolic::Expression max = symbolic::Expression::CreateNumericValue("max");
  cg->AppendLineOfCode(cg->assign("double max", "-INFINITY") + cg->linesep());

  std::string check_max_expression =
      cg->if_expr(generator_.I("i") + " > " + max.to_string()) + " {\n\t " +
      cg->assign("max", generator_.I("i")) + cg->linesep() + "\n}\n";

  cg->AppendLineOfCode(cg->for_loop(
      "size_t i = 0", "i < " + std::to_string(GetDimensions().num_inputs),
      "++i", check_max_expression));
  return max;
}

// Taken from the derivation here:
// https://stats.stackexchange.com/questions/235528/backpropagation-with-softmax-cross-entropy
void SoftmaxLayer::InputGradientCode(const symbolic::Expression &input_index,
                                     codegen::Generator *cg) const {
  symbolic::Expression max = AppendMaxCode(cg);
  symbolic::Expression output_k = GenerateOutputSymbol(input_index, max);
  symbolic::Expression retval(0.0);
  for (size_t i = 0; i < dimensions_.num_inputs; ++i) {
    symbolic::Expression output_i = GenerateOutputSymbol(i, max);
    symbolic::Expression kronecker = symbolic::KroneckerDelta(i, input_index);
    retval += ((output_i * (kronecker - output_k)) * Expression::CreateNumericValue(generator_.GRADIENT(i)));
  }
  cg->AppendLineOfCode("return " + retval.to_string() + cg->linesep());
}

void SoftmaxLayer::WeightGradientCode(const symbolic::Expression &weight_index,
                                      codegen::Generator *cg) const {
  cg->AppendLineOfCode("return 0.0" + cg->linesep());
}

std::unique_ptr<LayerImpl> SoftmaxLayer::Clone() const {
  return std::make_unique<SoftmaxLayer>(dimensions_.num_inputs, layer_index_);
}

}  // namespace nnet

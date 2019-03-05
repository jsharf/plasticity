#include "math/nnet/softmax_layer.h"
#include <memory>

namespace nnet {

// Max is subtracted from numerator and denominator in order to max softmax
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
  cg->AppendLineOfCode(cg->assign("double max", "0.0") + cg->linesep());

  std::string check_max_expression =
      cg->if_expr(generator_.I("i") + " > " + max.to_string()) + " {\n\t " +
      cg->assign("max", generator_.I("i")) + cg->linesep() + "\n}\n";

  cg->AppendLineOfCode(cg->for_loop(
      "size_t i = 0", "i < " + std::to_string(GetDimensions().num_inputs),
      "++i", check_max_expression));
  return max;
}

void SoftmaxLayer::InputGradientCode(const symbolic::Expression &input_index,
                                     codegen::Generator *cg) const {
  symbolic::Expression max = AppendMaxCode(cg);
  symbolic::Expression output = GenerateOutputSymbol(input_index, max);
  symbolic::Expression deriv =
      output.Derive(generator_.I(input_index).to_string());
  symbolic::Expression retval = generator_.GRADIENT(input_index) * deriv;

  /// DEBUG
  symbolic::Expression expsum = symbolic::NumericValue(0.0); 

  for (size_t i = 0; i < dimensions_.num_inputs; ++i) {
    expsum =
        expsum + symbolic::Exp(Expression::CreateNumericValue(generator_.I(i)) - max);
  }

  std::cout << "deriv: " <<  deriv.to_string() << std::endl;

  cg->AppendLineOfCode(";double expsum = " + expsum.to_string() + ";\nif ((expsum == 0) && (isnan(" + deriv.to_string() +
                       "))) { "
                       "for (int i = 0; i < " +
                       std::to_string(dimensions_.num_inputs) +
                       "; ++i) {"
                       "if (pow(2.71828, I[i]) < 0) {"
                       "printf(\"weird input causes neg pow: %f, res: %f\\n\", "
                       "I[i], pow(2.71828, I[i]));"
                       "} else if (fabs(pow(2.71828, I[i]) - 1) <= 0.01) {"
                         "printf(\"found zero, exp(zero) ->1\\n\");"
                       "}"
                       "}}");

  cg->AppendLineOfCode(
      "printf(\"exp(Max): %.10f, Denom: %.10f, Deriv: %.10f, Grad: "
      "%.10f\\n\", exp(max), " +
      expsum.to_string() + ", " + deriv.to_string() + ", " +
      generator_.GRADIENT(input_index).to_string() + ");");
  /// DEBUG
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

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

  /// DEBUG
//  symbolic::Expression expsum = symbolic::NumericValue(0.0);
//
//  for (size_t i = 0; i < dimensions_.num_inputs; ++i) {
//    expsum =
//        expsum + symbolic::Exp(Expression::CreateNumericValue(generator_.I(i)) - max);
//  }
//
//  cg->AppendLineOfCode(";double expsum = " + expsum.to_string() + ";\nif (!isfinite(1.0/" + expsum.to_string() +
//                       ")) { "
//                       "for (int i = 0; i < " +
//                       std::to_string(dimensions_.num_inputs) +
//                       "; ++i) {"
//                       "printf(\"weird input - max causes zero pow. max: %f, I[i]: %f, I[i] - max: %f res: %f\\n\", "
//                       "max, I[i], I[i] - max, pow(2.71828, I[i] - max));"
//                       "}}");
//
  //cg->AppendLineOfCode(
  //    "printf(\"exp(Max - Max): %.10f, Denom: %.10f, Deriv: %.10f, Grad: "
  //    "%.10f\\n\", exp(max-max), " +
  //    expsum.to_string() + ", " + deriv.to_string() + ", " +
  //    generator_.GRADIENT(input_index).to_string() + ");");
  /// DEBUG

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

void SoftmaxLayer::InputGradientCode(const symbolic::Expression &input_index,
                                     codegen::Generator *cg) const {
  symbolic::Expression max = AppendMaxCode(cg);
  symbolic::Expression output = GenerateOutputSymbol(input_index, max);
  symbolic::Expression deriv =
      output.Derive(generator_.I(input_index).to_string());
  symbolic::Expression retval = generator_.GRADIENT(input_index) * deriv;

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

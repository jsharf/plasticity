#include "math/nnet/softmax_layer.h"
#include <memory>

namespace nnet {

symbolic::Expression SoftmaxLayer::GenerateOutputCode(const symbolic::Expression& index) const {
  symbolic::Expression expsum = symbolic::NumericValue(0.0); 

  for (size_t i = 0; i < dimensions_.num_inputs; ++i) {
    expsum = expsum + symbolic::Exp(Expression::CreateNumericValue(generator_.I(i)));
  }

  return symbolic::Exp(symbolic::Expression::CreateNumericValue("I[" + index.to_string() + "]"))/expsum;
}

symbolic::Expression SoftmaxLayer::InputGradientCode(
    const symbolic::Expression& input_index) const {
  symbolic::Expression output =
      activation_function_(generator_.InputSymbolic(input_index));
  symbolic::Expression deriv =
      output.Derive(generator_.InputSymbolic(input_index).to_string());
  return generator_.GRADIENT(input_index) * deriv;
}

symbolic::Expression SoftmaxLayer::WeightGradientCode(
    const symbolic::Expression& weight_index) const {
  return symbolic::Expression(0.0);   
}

std::unique_ptr<LayerImpl> SoftmaxLayer::Clone() const {
  return std::make_unique<SoftmaxLayer>(dimensions_.num_inputs, layer_index_);
}

}  // namespace nnet

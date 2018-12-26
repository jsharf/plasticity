#include <memory>
#include "math/nnet/activation_layer.h"

namespace nnet {

symbolic::Expression ActivationLayer::GenerateOutputCode(
    const symbolic::Expression& index) const {
  return activation_function_(generator_.I(index));
}

symbolic::Expression ActivationLayer::InputGradientCode(
    const symbolic::Expression& input_index) const {
  symbolic::Expression output =
      activation_function_(generator_.I(input_index));
  symbolic::Expression deriv =
      output.Derive(generator_.I(input_index).to_string());
  return generator_.GRADIENT(input_index) * deriv;
}

symbolic::Expression ActivationLayer::WeightGradientCode(
    const symbolic::Expression& weight_index) const {
  return symbolic::Expression(0.0);   
}

std::unique_ptr<LayerImpl> ActivationLayer::Clone() const {
  return std::make_unique<ActivationLayer>(dimensions_.num_inputs,
                                           activation_function_, layer_index_);
}

}  // namespace nnet

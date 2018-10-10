#include "math/nnet/dense_layer.h"

#include <cassert>

namespace nnet {

DenseLayer::DenseLayer(const Dimensions& dimensions,
                       const ActivationFunctionType& activation_function,
                       size_t layer_index)
    : Super(dimensions, layer_index),
      generator_(dimensions),
      activation_function_(activation_function) {}

const std::vector<std::string>& DenseLayer::weights() const {
  return generator_.weights();
}

symbolic::Expression DenseLayer::GenerateOutputCode(
    const symbolic::Expression& output_index) const {
  symbolic::Expression sum = 0;
  for (size_t i = 0; i < dimensions_.num_inputs; ++i) {
    sum += generator_.BoundsCheckedW(output_index, Expression(i)) * generator_.I(i);
  }

  // Bias input.
  sum += generator_.W(output_index) * 1;

  return activation_function_(sum);
}

std::unique_ptr<LayerImpl> DenseLayer::Clone() const {
  return std::make_unique<DenseLayer>(dimensions_, activation_function_,
                                      layer_index_);
}

}  // namespace nnet

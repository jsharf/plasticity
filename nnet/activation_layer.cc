#include "math/nnet/activation_layer.h"
#include <memory>

namespace nnet {

symbolic::Expression ActivationLayer::GenerateOutputCode(
    const symbolic::Expression& index) const {
  return activation_function_(generator_.InputSymbolic(index));
}

std::unique_ptr<LayerImpl> ActivationLayer::Clone() const {
  return std::make_unique<ActivationLayer>(dimensions_.num_inputs, activation_function_, layer_index_);
}

}  // namespace nnet

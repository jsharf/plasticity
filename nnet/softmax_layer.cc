#include "math/nnet/softmax_layer.h"
#include <memory>

namespace nnet {

symbolic::Expression SoftmaxLayer::GenerateOutputCode(const symbolic::Expression& index) const {
  symbolic::Expression expsum = symbolic::NumericValue(0.0); 

  for (size_t i = 0; i < dimensions_.num_inputs; ++i) {
    expsum = expsum + Exp(generator_.I(i));
  }

  return Exp(symbolic::Expression::CreateNumericValue("I[" + index + "]"))/expsum;
}

std::unique_ptr<LayerImpl> SoftmaxLayer::Clone() const {
  return std::make_unique<SoftmaxLayer>(dimensions_.num_inputs, layer_index_);
}

}  // namespace nnet

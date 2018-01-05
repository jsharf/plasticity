#include "math/nnet/layer.h"

namespace nnet {

// Layer Class Implementation.

// Boilerplate constructors.
Layer::Layer(std::unique_ptr<LayerImpl>&& root) : impl_(std::move(root)) {}
Layer::Layer(Layer&& other) : impl_(std::move(other.impl_)) {}

// FeedForward layer static constructors.
Layer Layer::FeedForwardLayer(size_t layer_index, const Dimensions& dimensions,
                              const ActivationFunctionType& activation_function,
                              SymbolGenerator* generator) {
  return Layer(std::make_unique<FeedForwardLayer>(
      dimensions, activation_function, generator, layer_index));
}

Layer Layer::FeedForwardLayer(size_t layer_index, const Dimensions& dimensions,
                              SymbolGenerator* generator) {
  //return Layer(
  //    std::make_unique<FeedForwardLayer>(dimensions, generator, layer_index));
  return Layer();
}

// Convolution layer static constructor.
Layer Layer::ConvolutionLayer(size_t layer_index,
                              const VolumeDimensions& dimensions,
                              const FilterParams& params,
                              SymbolGenerator* generator) {
  //return Layer(std::make_unique<ConvolutionLayer>(dimensions, params, generator,
  //                                                layer_index));
  return Layer();
}

Layer::WeightArray Layer::weights() { return impl_->weights(); }

Matrix<symbolic::Expression> Layer::GenerateExpression(
    const Matrix<symbolic::Expression>& input) {
  return impl_->GenerateExpression(input);
}

stats::Normal Layer::XavierInitializer() { return impl_->XavierInitializer(); }

}  // namespace nnet

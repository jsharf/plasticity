#include "math/nnet/layer.h"

namespace nnet {

// Layer Class Implementation.

// Boilerplate constructors.
Layer::Layer(std::unique_ptr<LayerImpl> root) : impl_(std::move(root)) {}
Layer::Layer(Layer&& other) : impl_(std::move(other.impl_)) {}
Layer::Layer(const Layer& other) : impl_(other.impl_->Clone()) {}

// FeedForward layer static constructors.
Layer Layer::MakeFeedForwardLayer(
    size_t layer_index, const Dimensions& dimensions,
    const ActivationFunctionType& activation_function,
    SymbolGenerator* generator) {
  return Layer(std::make_unique<FeedForwardLayer>(
      dimensions, activation_function, generator, layer_index));
}

Layer Layer::MakeFeedForwardLayer(size_t layer_index,
                                  const Dimensions& dimensions,
                                  SymbolGenerator* generator) {
  return Layer(
      std::make_unique<FeedForwardLayer>(dimensions, generator, layer_index));
}

// Convolution layer static constructor.
Layer Layer::MakeConvolutionLayer(size_t layer_index,
                                  const VolumeDimensions& dimensions,
                                  const FilterParams& params,
                                  SymbolGenerator* generator) {
  return Layer(std::make_unique<ConvolutionLayer>(dimensions, params, generator,
                                                  layer_index));
}

Layer Layer::MakeMaxPoolLayer(size_t layer_index, const VolumeDimensions& input,
                              const AreaDimensions& output,
                              SymbolGenerator* generator) {
  return Layer(
      std::make_unique<MaxPoolLayer>(input, output, generator, layer_index));
}

Layer Layer::MakeActivationLayer(
    size_t layer_index, size_t size,
    const ActivationFunctionType& activation_function,
    SymbolGenerator* generator) {
  return Layer(std::make_unique<ActivationLayer>(size, activation_function,
                                                 generator, layer_index));
}

Layer::WeightArray Layer::weights() { return impl_->weights(); }

Matrix<symbolic::Expression> Layer::GenerateExpression(
    const Matrix<symbolic::Expression>& input) {
  return impl_->GenerateExpression(input);
}

stats::Normal Layer::XavierInitializer() { return impl_->XavierInitializer(); }

}  // namespace nnet

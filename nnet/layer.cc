#include "math/nnet/layer.h"

#include <fstream>

namespace nnet {

// Layer Class Implementation.

// Boilerplate constructors.
Layer::Layer(std::unique_ptr<LayerImpl>&& root) : impl_(std::move(root)) {}
Layer::Layer(Layer&& other) : impl_(std::move(other.impl_)) {}
Layer::Layer(const Layer& other) : impl_(other.impl_->Clone()) {}

// Boilerplate operator.
Layer& Layer::operator=(const Layer& rhs) {
  impl_ = rhs.impl_->Clone();
  return *this;
}
Layer& Layer::operator=(Layer&& rhs) {
  impl_ = std::move(rhs.impl_);
  return *this;
}

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

Layer Layer::MakeSoftmaxLayer(size_t layer_index, size_t size,
                              SymbolGenerator* generator) {
  return Layer(std::make_unique<SoftmaxLayer>(size, generator, layer_index));
}

Layer::WeightArray Layer::weights() const { return impl_->weights(); }

Matrix<symbolic::Expression> Layer::GenerateExpression(
    const Matrix<symbolic::Expression>& input) {
  return impl_->GenerateExpression(input);
}

stats::Normal Layer::XavierInitializer() { return impl_->XavierInitializer(); }

namespace {

// Replaces the first instance of find with replace. Returns true if edit made.
bool FindAndReplace(std::string* text, std::string find, std::string replace) {
  size_t location = text->find(find);
  if (location == std::string::npos) {
    return false;
  }
  text->replace(location, find.size(), replace);
  return true;
}

}  // namespace

// TODO teardown symbolgenerator concept. Make this const again. :/.
std::string Layer::GenerateEvaluationKernel(const Matrix<symbolic::Expression>& input) {
  std::ifstream evaluate_file("math/nnet/kernels/evaluate.kernel.cl");
  std::stringstream buffer;
  buffer << evaluate_file.rdbuf();
  std::string evaluate_source = buffer.str();

  Matrix<symbolic::Expression> output_expressions = GenerateExpression(input);

  std::stringstream outputs;
  for (size_t i = 0; i < GetDimensions().num_outputs; ++i) {
    outputs << "case " << i << ":" << std::endl;
    outputs << "  return (" << output_expressions.at(i, 0).to_string() << ");"
            << std::endl;
  }

  if (!FindAndReplace(&evaluate_source, "EXPRESSION_HERE", outputs.str())) {
    std::cerr << "Could not find template substring \"EXPRESSION_HERE\"." << std::endl;
    std::exit(1);
  }

  if (!FindAndReplace(&evaluate_source, "LAYERID",
                      std::to_string(impl_->layer_index()))) {
    std::cerr << "Could not find template substring \"LAYERID\"." << std::endl;
    std::exit(1);
  }

  return evaluate_source;
}

}  // namespace nnet

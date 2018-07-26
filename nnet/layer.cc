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
    const ActivationFunctionType& activation_function) {
  return Layer(std::make_unique<FeedForwardLayer>(
      dimensions, activation_function, layer_index));
}

Layer Layer::MakeFeedForwardLayer(size_t layer_index,
                                  const Dimensions& dimensions) {
  return Layer(std::make_unique<FeedForwardLayer>(dimensions, layer_index));
}

// Convolution layer static constructor.
Layer Layer::MakeConvolutionLayer(size_t layer_index,
                                  const VolumeDimensions& dimensions,
                                  const FilterParams& params) {
  return Layer(
      std::make_unique<ConvolutionLayer>(dimensions, params, layer_index));
}

Layer Layer::MakeMaxPoolLayer(size_t layer_index, const VolumeDimensions& input,
                              const AreaDimensions& output) {
  return Layer(std::make_unique<MaxPoolLayer>(input, output, layer_index));
}

Layer Layer::MakeActivationLayer(
    size_t layer_index, size_t size,
    const ActivationFunctionType& activation_function) {
  return Layer(std::make_unique<ActivationLayer>(size, activation_function,
                                                 layer_index));
}

Layer Layer::MakeSoftmaxLayer(size_t layer_index, size_t size
    ) {
  return Layer(std::make_unique<SoftmaxLayer>(size, layer_index));
}

Layer::WeightArray Layer::weights() const { return impl_->weights(); }

Matrix<symbolic::Expression> Layer::GenerateExpression() {
  Matrix<symbolic::Expression> input = InputExpression();
  return impl_->GenerateExpression(input);
}

stats::Normal Layer::XavierInitializer() const {
  return stats::Normal(0, 1.0 / (impl_->GetDimensions().num_inputs));
}

void Layer::XavierInitializeWeights() {
  if (weights().size() == 0) {
    return;
  }

  stats::Normal X = XavierInitializer();
  for (const std::string& weight : weights()) {
    env()[weight].real() = X.sample();
  }
}

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

std::string FileToString(std::string filepath) {
  std::ifstream input(filepath);
  std::stringstream buffer;
  buffer << input.rdbuf();
  return buffer.str();
}

}  // namespace

std::string Layer::GenerateEvaluationKernel() {
  std::string evaluate_source =
      FileToString("math/nnet/kernels/evaluate.kernel.cl");

  Matrix<symbolic::Expression> input = InputExpression();
  size_t rows = std::get<0>(input.size());
  size_t cols = std::get<1>(input.size());

  // Validate input dimensions.
  if ((rows != impl_->GetDimensions().num_inputs) || (cols != 1)) {
    std::cerr << "Error: Layer::GenerateEvaluationKernel() called on input "
                 "of incorrect size: "
              << "(" << rows << ", " << cols << ")" << std::endl;
    std::exit(1);
  }

  Matrix<symbolic::Expression> output_expressions = GenerateExpression();

  std::stringstream outputs;
  for (size_t i = 0; i < GetDimensions().num_outputs; ++i) {
    outputs << "case " << i << ":" << std::endl;
    outputs << "  return (" << output_expressions.at(i, 0).to_string() << ");"
            << std::endl;
  }

  if (!FindAndReplace(&evaluate_source, "EXPRESSION_HERE", outputs.str())) {
    std::cerr << "Could not find template substring \"EXPRESSION_HERE\"."
              << std::endl;
    std::exit(1);
  }

  // LAYERID shows up exactly 3 times in the kernel template. Yes this is hacky.
  // It's a side project.
  // TODO(sharf): Use a real templating system for the kernels.
  const size_t kNumberExpectedReplacements = 3;

  for (size_t replacement_time = 0;
       replacement_time < kNumberExpectedReplacements; ++replacement_time) {
    if (!FindAndReplace(&evaluate_source, "LAYERID",
                        std::to_string(impl_->layer_index()))) {
      std::cerr << "Could not find template substring \"LAYERID\"."
                << std::endl;
      std::exit(1);
    }
  }

  return evaluate_source;
}

std::string Layer::WeightsToString() const {
  std::stringstream output;
  output << "layer_" << impl_->layer_index() << ": {" << std::endl;
  for (const auto& kv : env_) {
    output << kv.first << ":" << kv.second.to_string() << "," << std::endl;
  }
  output << "}" << std::endl;
  return output.str();
}

Matrix<symbolic::Expression> Layer::BackpropGradients() const {
  Matrix<symbolic::Expression> bp_gradients(impl_->GetDimensions().num_outputs,
                                            1);
  for (size_t output_num = 0; output_num < bp_gradients.dimensions().rows;
       ++output_num) {
    bp_gradients.at(output_num, 0) =
        symbolic::Expression(generator_.GRADIENT(output_num));
  }
  return bp_gradients;
}

Matrix<symbolic::Expression> Layer::InputGradients() {
  Matrix<symbolic::Expression> input = InputExpression();
  Matrix<symbolic::Expression> output_expressions = GenerateExpression();
  Matrix<symbolic::Expression> bp_gradients = BackpropGradients();

  size_t rows = std::get<0>(input.size());

  // For each input... calculate the propagated differential expression.
  Matrix<symbolic::Expression> input_gradient_expressions(rows, 1);
  for (size_t i = 0; i < rows; ++i) {
    Matrix<symbolic::Expression> output_deriv_expressions =
        output_expressions.Map(
            std::function<symbolic::Expression(const symbolic::Expression&)>(
                [i, input](
                    const symbolic::Expression& exp) -> symbolic::Expression {
                  return exp.Derive(input.at(i, 0).to_string());
                }));
    Matrix<symbolic::Expression> input_gradient_expression =
        bp_gradients.Transpose() * output_deriv_expressions;
    if ((input_gradient_expression.dimensions().rows != 1) ||
        (input_gradient_expression.dimensions().cols != 1)) {
      std::cerr << "Error: invalid dimension of result mat mult." << std::endl;
      std::exit(1);
    }
    input_gradient_expressions.at(i, 0) = input_gradient_expression.at(0, 0);
  }
  return input_gradient_expressions;
}

Matrix<symbolic::Expression> Layer::WeightGradients() {
  Matrix<symbolic::Expression> input = InputExpression();
  Matrix<symbolic::Expression> output_expressions = GenerateExpression();
  Matrix<symbolic::Expression> bp_gradients = BackpropGradients();

  Matrix<symbolic::Expression> weight_gradient_expressions(weights().size(), 1);
  for (size_t i = 0; i < weights().size(); ++i) {
    Matrix<symbolic::Expression> output_deriv_expressions =
        output_expressions.Map(
            std::function<symbolic::Expression(const symbolic::Expression&)>(
                [i, this](const symbolic::Expression& exp) {
                  return exp.Derive(weights()[i]);
                }));
    Matrix<symbolic::Expression> weight_gradient_expression =
        bp_gradients.Transpose() * output_deriv_expressions;
    if ((std::get<0>(weight_gradient_expression.size()) != 1) ||
        (std::get<1>(weight_gradient_expression.size()) != 1)) {
      std::cerr << "Error: invalid dimension of result mat mult." << std::endl;
      std::exit(1);
    }
    weight_gradient_expressions.at(i, 0) = weight_gradient_expression.at(0, 0);
  }
  return weight_gradient_expressions;
}

std::string Layer::GenerateTrainingKernels() {
  std::string train_source =
      FileToString("math/nnet/kernels/back_prop.kernel.cl");

  Matrix<symbolic::Expression> input_gradient_expressions = InputGradients();

  std::stringstream input_gradients;
  for (size_t i = 0; i < GetDimensions().num_inputs; ++i) {
    input_gradients << "case " << i << ":" << std::endl;
    input_gradients << "  return ("
                    << input_gradient_expressions.at(i, 0).to_string() << ");"
                    << std::endl;
  }

  Matrix<symbolic::Expression> weight_gradient_expressions = WeightGradients();

  std::stringstream weight_gradients;
  for (size_t i = 0; i < weights().size(); ++i) {
    weight_gradients << "case " << i << ":" << std::endl;
    weight_gradients << "  return ("
                     << weight_gradient_expressions.at(i, 0).to_string() << ");"
                     << std::endl;
  }

  if (!FindAndReplace(&train_source, "INPUT_GRADIENTS_HERE",
                      input_gradients.str())) {
    std::cerr << "Could not find template substring \"INPUT_GRADIENTS_HERE\"."
              << std::endl;
    std::exit(1);
  }

  if (!FindAndReplace(&train_source, "WEIGHT_GRADIENTS_HERE",
                      weight_gradients.str())) {
    std::cerr << "Could not find template substring \"WEIGHT_GRADIENTS_HERE\"."
              << std::endl;
    std::exit(1);
  }

  // Yes this is hacky.  It's a side project.
  // TODO(sharf): Use a real templating system for the kernels.
  const size_t kNumberExpectedReplacements = 6;

  for (size_t replacement_time = 0;
       replacement_time < kNumberExpectedReplacements; ++replacement_time) {
    if (!FindAndReplace(&train_source, "LAYERID",
                        std::to_string(impl_->layer_index()))) {
      std::cerr << "Could not find template substring \"LAYERID\"."
                << std::endl;
      std::exit(1);
    }
  }

  return train_source;
}

Matrix<symbolic::Expression> Layer::InputExpression() const {
  const size_t num_inputs = GetDimensions().num_inputs;
  Matrix<symbolic::Expression> result(num_inputs, 1);
  for (size_t i = 0; i < num_inputs; ++i) {
    result.at(i, 0) = symbolic::CreateExpression(symbol_generator()->I(i));
  }
  return result;
}

Matrix<symbolic::Expression> Layer::OutputExpression() const {
  const size_t num_outputs = GetDimensions().num_outputs;
  Matrix<symbolic::Expression> result(num_outputs, 1);
  for (size_t i = 0; i < num_outputs; ++i) {
    result.at(i, 0) = symbolic::CreateExpression(symbol_generator()->O(i));
  }
  return result;
}

}  // namespace nnet

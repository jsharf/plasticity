#include "nnet/layer.h"
#include "nnet/nnet.h"

#include <fstream>
#include <future>
#include <limits>
#include <thread>

namespace nnet {

using symbolic::Expression;

namespace internal {

namespace kernel_symbols {

constexpr char kOutputIndex[] = "output_index";
constexpr char kInputIndex[] = "input_index";
constexpr char kWeightIndex[] = "weight_index";

}  // namespace kernel_symbols

}  // namespace nnet

// Layer Class Implementation.

// Boilerplate constructors.
Layer::Layer(std::unique_ptr<LayerImpl> &&root)
    : impl_(std::move(root)),
      eval_workgroup_size_(
          CalculateWorkgroupSize(impl_->GetDimensions().num_outputs)),
      weight_train_workgroup_size_(
          CalculateWorkgroupSize(impl_->weights().size())),
      bp_train_workgroup_size_(
          CalculateWorkgroupSize(impl_->GetDimensions().num_inputs)),
      weights_(impl_->weights().size()) {}
Layer::Layer(Layer &&other)
    : impl_(std::move(other.impl_)),
      eval_workgroup_size_(other.eval_workgroup_size_),
      weight_train_workgroup_size_(other.weight_train_workgroup_size_),
      bp_train_workgroup_size_(other.bp_train_workgroup_size_),
      weights_(std::move(other.weights_)) {}
Layer::Layer(const Layer &other)
    : impl_(other.impl_->Clone()),
      eval_workgroup_size_(other.eval_workgroup_size_),
      weight_train_workgroup_size_(other.weight_train_workgroup_size_),
      bp_train_workgroup_size_(other.bp_train_workgroup_size_),
      weights_(other.weights_) {}

void Layer::RegisterToNetwork(nnet::Nnet *network) {
  // Awesome, we have a network registered. Use it to allocate a buffer for the
  // weights.
  nnet_ = network;
  nnet_->RegisterBuffer(&weights_);
}

// Dense layer static constructors.
Layer Layer::MakeDenseLayer(size_t layer_index, const Dimensions &dimensions) {
  return Layer(std::make_unique<DenseLayer>(dimensions, layer_index));
}

// Convolution layer static constructor.
Layer Layer::MakeConvolutionLayer(size_t layer_index,
                                  const VolumeDimensions &dimensions,
                                  const FilterParams &params) {
  return Layer(
      std::make_unique<ConvolutionLayer>(dimensions, params, layer_index));
}

Layer Layer::MakeMaxPoolLayer(size_t layer_index, const VolumeDimensions &input,
                              const AreaDimensions &output) {
  return Layer(std::make_unique<MaxPoolLayer>(input, output, layer_index));
}

Layer Layer::MakeActivationLayer(
    size_t layer_index, size_t size,
    const ActivationFunctionType &activation_function) {
  return Layer(std::make_unique<ActivationLayer>(size, activation_function,
                                                 layer_index));
}

Layer Layer::MakeSoftmaxLayer(size_t layer_index, size_t size) {
  return Layer(std::make_unique<SoftmaxLayer>(size, layer_index));
}

stats::Normal Layer::XavierInitializer() const {
  return stats::Normal(0, 1.0 / (impl_->GetDimensions().num_inputs));
}

void Layer::XavierInitializeWeights() {
  if (weights_.size() == 0) {
    return;
  }

  weights_.MoveToCpu();
  stats::Normal X = XavierInitializer();
  for (size_t i = 0; i < weights_.size(); ++i) {
    weights_[i] = X.sample();
  }
}

void Layer::InitializeWeights(double value) {
  weights_.MoveToCpu();
  for (size_t i = 0; i < weights_.size(); ++i) {
    weights_[i] = value;
  }
}

namespace {

// Replaces the first instance of find with replace. Returns true if edit made.
bool FindAndReplace(std::string *text, std::string find, std::string replace) {
  size_t location = text->find(find);
  if (location == std::string::npos) {
    return false;
  }
  text->replace(location, find.size(), replace);
  return true;
}

std::string FileToString(std::string filepath) {
  std::ifstream test(filepath);
  if (!test.is_open()) {
    // This is a hack due to Bazel's broken handling of external dependencies.
    // In plasticity, the kernel source files show up with filepaths WRT
    // project runfiles. But in external dependencies, they show up under
    // "external/".
    filepath = "external/plasticity/" + filepath;
  }
  std::ifstream input(filepath);
  if (!input.is_open()) {
  }
  std::stringstream buffer;
  buffer << input.rdbuf();
  return buffer.str();
}

}  // namespace

std::string Layer::GenerateEvaluationKernel() const {
  std::string evaluate_source =
      FileToString("nnet/kernels/evaluate.kernel.cl");

  // Validate input dimensions.
  if ((GetDimensions().num_inputs == 0) || (GetDimensions().num_outputs == 0)) {
    std::cerr
        << "Error: Layer::GenerateEvaluationKernel() called on dimensions "
           "of incorrect size: "
        << "(inputs: " << GetDimensions().num_inputs
        << ", outputs: " << GetDimensions().num_outputs << ")" << std::endl;
    std::cerr << "Layer: " << LayerSuffix() << std::endl;
    std::exit(1);
  }

  codegen::CudaGenerator generator;
  impl_->GenerateOutputCode(
      Expression::CreateInteger(internal::kernel_symbols::kOutputIndex),
      &generator);

  if (!FindAndReplace(&evaluate_source, "EXPRESSION_HERE", generator.code())) {
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

std::string Layer::WeightsToString() {
  weights_.MoveToCpu();
  std::stringstream output;
  output.precision(std::numeric_limits<double>::max_digits10);
  output << "weights:" << "\": [" << std::endl;
  for (size_t i = 0; i < weights_.size(); ++i) {
    output << weights_[i];
    if (i != weights_.size() - 1) {
      output << ",";
    }
    output << std::endl;
  }
  output << "]" << std::endl;
  return output.str();
}

std::string Layer::GenerateTrainingKernels() const {
  std::string train_source =
      FileToString("nnet/kernels/back_prop.kernel.cl");

  codegen::CudaGenerator input_gen;
  impl_->InputGradientCode(
      Expression::CreateInteger(internal::kernel_symbols::kInputIndex),
      &input_gen);

  codegen::CudaGenerator weight_gen;
  impl_->WeightGradientCode(
      Expression::CreateInteger(internal::kernel_symbols::kWeightIndex),
      &weight_gen);

  if (!FindAndReplace(&train_source, "INPUT_GRADIENTS_HERE",
                      input_gen.code())) {
    std::cerr << "Could not find template substring \"INPUT_GRADIENTS_HERE\"."
              << std::endl;
    std::exit(1);
  }

  if (!FindAndReplace(&train_source, "WEIGHT_GRADIENTS_HERE",
                      weight_gen.code())) {
    std::cerr << "Could not find template substring \"WEIGHT_GRADIENTS_HERE\"."
              << std::endl;
    std::exit(1);
  }

  // Yes this is hacky.  It's a side project.
  // TODO(sharf): Use a real templating system for the kernels.
  while (true) {
    if (!FindAndReplace(&train_source, "LAYERID", LayerSuffix())) {
      // Done replacing all cases of LAYERID
      break;
    }
  }

  return train_source;
}

Matrix<Expression> Layer::InputExpression() const {
  const size_t num_inputs = GetDimensions().num_inputs;
  Matrix<Expression> result(num_inputs, 1);
  for (size_t i = 0; i < num_inputs; ++i) {
    result.at(i, 0) = Expression::CreateNumericValue(generator_.I(i));
  }
  return result;
}

Matrix<Expression> Layer::OutputExpression() const {
  const size_t num_outputs = GetDimensions().num_outputs;
  Matrix<Expression> result(num_outputs, 1);
  for (size_t i = 0; i < num_outputs; ++i) {
    result.at(i, 0) = Expression::CreateNumericValue(generator_.O(i));
  }
  return result;
}

}  // namespace nnet

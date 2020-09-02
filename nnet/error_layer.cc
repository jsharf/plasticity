#include "nnet/error_layer.h"

#include <fstream>

namespace nnet {

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
  std::stringstream buffer;
  buffer << input.rdbuf();
  return buffer.str();
}

std::string ErrorLayer::GenerateErrorKernels() const {
  std::string error_source = FileToString("nnet/kernels/error.kernel.cl");

  symbolic::Expression error = GenerateErrorComponent();
  symbolic::Expression gradient = error.Derive(O().to_string());

  if (!FindAndReplace(&error_source, "ERROR_EXPRESSION_HERE",
                      error.to_string())) {
    std::cerr << "Could not find template substring \"ERROR_EXPRESSION_HERE\"."
              << std::endl;
    std::exit(1);
  }
  if (!FindAndReplace(&error_source, "GRADIENT_EXPRESSION_HERE",
                      gradient.to_string())) {
    std::cerr
        << "Could not find template substring \"GRADIENT_EXPRESSION_HERE\"."
        << std::endl;
    std::exit(1);
  }
  return error_source;
}

constexpr char kIndexName[] = "index";

symbolic::Expression ErrorLayer::O() const {
  return symbolic::Expression::CreateNumericValue(
      "O[" + std::string(kIndexName) + "]");
}
symbolic::Expression ErrorLayer::E() const {
  return symbolic::Expression::CreateNumericValue(
      "E[" + std::string(kIndexName) + "]");
}

symbolic::Expression ErrorLayer::GenerateErrorComponent() const {
  switch (loss_function_) {
    case MeanSquared:
      return GenerateMseErrorComponent();
    case CrossEntropy:
      return GenerateCrossEntropyErrorComponent();
    default:
      std::cerr << "Error: Unknown loss function selected." << std::endl;
      std::exit(1);
  }
}

symbolic::Expression ErrorLayer::GenerateMseErrorComponent() const {
  symbolic::Expression error = (E() - O());
  return (error * error) / 2;
}

symbolic::Expression ErrorLayer::GenerateCrossEntropyErrorComponent() const {
  symbolic::Expression e = E();
  symbolic::Expression a = O();
  // Should be SafeLog?
  return symbolic::Expression(-1.0) * (e * symbolic::SafeLog(a));
}

}  // namespace nnet

#include "math/nnet/activation_layer.h"
#include <memory>

namespace nnet {

Matrix<symbolic::Expression> ActivationLayer::GenerateExpression(const Matrix<symbolic::Expression>& input) {
  auto dim = input.size();
  size_t rows = std::get<0>(dim);
  size_t cols = std::get<1>(dim);
  if ((rows != dimensions_.num_inputs) || (cols != 1)) {
    std::cerr << "Error: ActivationLayer::GenerateExpression called on input "
                 "of incorrect size: "
              << "(" << rows << ", " << cols << ")" << std::endl;
    std::exit(1);
  }

  Matrix<symbolic::Expression> output(rows, cols);

  for (size_t r = 0; r < rows; ++r) {
    output.at(r, 0) = activation_function_(input.at(r, 0));
  }

  return output;
}

std::unique_ptr<LayerImpl> ActivationLayer::Clone() const {
  return std::make_unique<ActivationLayer>(dimensions_.num_inputs, activation_function_, generator_, layer_index_);
}

}  // namespace nnet
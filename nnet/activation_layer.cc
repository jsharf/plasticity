#include "math/nnet/activation_layer.h"

namespace nnet {

Matrix<symbolic::Expression> GenerateExpression(const Matrix<symbolic::Expression>& input) {
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

}  // namespace nnet

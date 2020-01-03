#include "plasticity/geometry/dynamic_matrix.h"
#include "plasticity/stats/normal.h"
#include "plasticity/symbolic/expression.h"
#include "plasticity/symbolic/symbolic_util.h"

namespace nnet {

// Adds a bias input to the end of a column vector.
Matrix<symbolic::Expression> AddBias(Matrix<symbolic::Expression> x) {
  auto dim = x.size();
  size_t rows = std::get<0>(dim);
  size_t cols = std::get<1>(dim);
  if (cols != 1) {
    std::cerr << "Err: AddBias must only be called on column vectors!"
              << std::endl;
    std::exit(1);
  }
  Matrix<symbolic::Expression> biased_layer(std::get<0>(dim) + 1, 1);
  for (size_t i = 0; i < rows; ++i) {
    biased_layer.at(i, 0) = x.at(i, 0);
  }
  // Bias is always 1.
  biased_layer.at(rows, 0) = symbolic::CreateExpression("1");
  return biased_layer;
}

}

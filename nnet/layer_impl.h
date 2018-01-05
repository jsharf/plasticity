#ifndef LAYER_IMPL_H
#define LAYER_IMPL_H

#include "math/geometry/dynamic_matrix.h"
#include "math/stats/normal.h"
#include "math/symbolic/expression.h"
#include "math/symbolic/symbolic_util.h"

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


class SymbolGenerator {
 public:
  // Feed-forward layer weights.
  virtual std::string W(size_t layer, size_t node, size_t edge) = 0;
  // Convolution layer weights.
  virtual std::string W(size_t layer, size_t filter, size_t x, size_t y,
                        size_t z) = 0;
  // Convolution layer bias weights.
  virtual std::string W(size_t layer, size_t filter) = 0;

  virtual std::string I(size_t i) const = 0;
  virtual std::string O(size_t i) const = 0;
  virtual ~SymbolGenerator() {}
};

class LayerImpl {
 public:
  struct Dimensions {
    size_t num_inputs;
    size_t num_outputs;
  };

  // Dim(num_outputs * (num_inputs + 1))
  using WeightArray = std::vector<std::string>;

  using ActivationFunctionType =
      std::function<symbolic::Expression(const symbolic::Expression&)>;

  virtual WeightArray weights() = 0;
  virtual Matrix<symbolic::Expression> GenerateExpression(
      const Matrix<symbolic::Expression>& input) = 0;
  virtual stats::Normal XavierInitializer() const = 0;
  Dimensions GetDimensions() const { return dimensions_; }
  virtual ~LayerImpl() {}

 protected:
  LayerImpl(const Dimensions& dimensions, SymbolGenerator* generator,
            size_t layer_index)
      : dimensions_(dimensions),
        generator_(generator),
        layer_index_(layer_index) {}

  Dimensions dimensions_;
  SymbolGenerator* generator_;
  size_t layer_index_;
};

}  // namespace nnet

#endif /* LAYER_IMPL_H */

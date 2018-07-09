#ifndef LAYER_IMPL_H
#define LAYER_IMPL_H

#include "math/geometry/dynamic_matrix.h"
#include "math/stats/normal.h"
#include "math/symbolic/expression.h"
#include "math/symbolic/symbolic_util.h"

namespace nnet {

// Adds a bias input to the end of a column vector.
Matrix<symbolic::Expression> AddBias(Matrix<symbolic::Expression> x);

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

  // Residual gradients for back propagation.
  virtual symbolic::Expression GRADIENT(size_t i) const {
    return symbolic::Expression("GRADIENT[" + std::to_string(i) + "]");
  }

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

  virtual WeightArray weights() const = 0;
  virtual Matrix<symbolic::Expression> GenerateExpression(
      const Matrix<symbolic::Expression>& input) = 0;
  virtual stats::Normal XavierInitializer() const = 0;
  Dimensions GetDimensions() const { return dimensions_; }
  virtual std::unique_ptr<LayerImpl> Clone() const = 0;

  size_t layer_index() const { return layer_index_; }

  // Tread carefully... If you accidentally assign the wrong symbol generator to
  // a layer, you can end up in really weird hard to debug states.
  void SetSymbolGenerator(SymbolGenerator* generator) {
    generator_ = generator;
  }

  SymbolGenerator* symbol_generator() const {
    return generator_;
  }

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

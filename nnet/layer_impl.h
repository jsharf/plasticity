#ifndef LAYER_IMPL_H
#define LAYER_IMPL_H

#include "math/geometry/dynamic_matrix.h"
#include "math/nnet/symbol_generator.h"
#include "math/stats/normal.h"
#include "math/symbolic/expression.h"
#include "math/symbolic/symbolic_util.h"

namespace nnet {

// Adds a bias input to the end of a column vector.
Matrix<symbolic::Expression> AddBias(Matrix<symbolic::Expression> x);

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

  SymbolGenerator* symbol_generator() const { return generator_; }

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

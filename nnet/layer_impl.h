#ifndef LAYER_IMPL_H
#define LAYER_IMPL_H

#include "math/geometry/dynamic_matrix.h"
#include "math/nnet/layer_dimensions.h"
#include "math/stats/normal.h"
#include "math/symbolic/expression.h"
#include "math/symbolic/symbolic_util.h"

namespace nnet {

// Adds a bias input to the end of a column vector.
Matrix<symbolic::Expression> AddBias(Matrix<symbolic::Expression> x);

class LayerImpl {
 public:
  // Dim(num_outputs * (num_inputs + 1))
  // TODO(sharf): use std::vector<std::string> instead and rename weights() to
  // weightnames();
  using ActivationFunctionType =
      std::function<symbolic::Expression(const symbolic::Expression&)>;

  virtual const std::vector<std::string>& weights() const {
    // Default implementation.
    static const std::vector<std::string> *const empty = new std::vector<std::string>();
    return *empty;
  }

  virtual Matrix<symbolic::Expression> GenerateExpression(
      const Matrix<symbolic::Expression>& input) = 0;

  virtual std::unique_ptr<LayerImpl> Clone() const = 0;

  Dimensions GetDimensions() const { return dimensions_; }

  size_t layer_index() const { return layer_index_; }

  virtual ~LayerImpl() {}

 protected:
  LayerImpl(const Dimensions& dimensions, size_t layer_index)
      : dimensions_(dimensions), layer_index_(layer_index) {}

  Dimensions dimensions_;
  size_t layer_index_;
};

}  // namespace nnet

#endif /* LAYER_IMPL_H */

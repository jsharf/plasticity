#ifndef LAYER_IMPL_H
#define LAYER_IMPL_H

#include "math/codegen/codegen.h"
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
    static const std::vector<std::string> empty = std::vector<std::string>();
    return empty;
  }

  virtual codegen::CudaGenerator GenerateOutputCode(
      const symbolic::Expression& output_index) const = 0;

  virtual codegen::CudaGenerator InputGradientCode(
      const symbolic::Expression& input_index) const = 0;

  virtual codegen::CudaGenerator WeightGradientCode(
      const symbolic::Expression& weight_index) const = 0;

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

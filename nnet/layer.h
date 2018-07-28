#ifndef LAYER_H
#define LAYER_H
#include "math/geometry/dynamic_matrix.h"
#include "math/nnet/activation_layer.h"
#include "math/nnet/convolution_layer.h"
#include "math/nnet/feed_forward_layer.h"
#include "math/nnet/layer_impl.h"
#include "math/nnet/max_pool_layer.h"
#include "math/nnet/softmax_layer.h"
#include "math/stats/normal.h"
#include "math/symbolic/expression.h"
#include "math/symbolic/symbolic_util.h"

#include <array>
#include <cassert>
#include <iterator>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

namespace nnet {

// Holds a pointer to a Layer and manages the resources.
class Layer {
 public:
  // Public exported types
  using WeightArray = LayerImpl::WeightArray;
  using ActivationFunctionType = LayerImpl::ActivationFunctionType;

  // Constructors.
  Layer() {
    // Default to zero initialization. For Xavier initialization, call
    // XavierInitializeWeights().
    for (const std::string& weight : weights()) {
      env()[weight].real() = 0;
    }
  }
  Layer(std::unique_ptr<LayerImpl>&& root);
  explicit Layer(Layer&& rhs);
  Layer(const Layer& rhs);

  // Destructor.
  virtual ~Layer() {}

  // Assignment Operators.
  Layer& operator=(const Layer& rhs);
  Layer& operator=(Layer&& rhs);

  // FeedForward Layer constructors.
  static Layer MakeFeedForwardLayer(
      size_t layer_index, const Dimensions& dimensions,
      const ActivationFunctionType& activation_function);

  static Layer MakeFeedForwardLayer(size_t layer_index,
                                    const Dimensions& dimensions);

  // Convolutional Layer constructors.
  static Layer MakeConvolutionLayer(size_t layer_index,
                                    const VolumeDimensions& dimensions,
                                    const FilterParams& params);

  static Layer MakeActivationLayer(
      size_t layer_index, size_t size,
      const ActivationFunctionType& activation_function);

  static Layer MakeSoftmaxLayer(size_t layer_index, size_t size);

  static Layer MakeMaxPoolLayer(size_t layer_index,
                                const VolumeDimensions& input,
                                const AreaDimensions& output);

  WeightArray weights() const;

  symbolic::Environment& env() { return env_; }
  const symbolic::Environment& env() const { return env_; }

  std::string WeightsToString() const;

  Matrix<symbolic::Expression> BackpropGradients() const;
  Matrix<symbolic::Expression> InputGradients();
  Matrix<symbolic::Expression> WeightGradients();

  Matrix<symbolic::Expression> GenerateExpression();

  stats::Normal XavierInitializer() const;
  void XavierInitializeWeights();
  
  Dimensions GetDimensions() const { return impl_->GetDimensions(); }

  // This function returns the source code of an OpenCL kernel which evaluates
  // the output of this layer, given the input.
  std::string GenerateEvaluationKernel();

  std::string EvaluateKernelName() const {
    return "evaluate_" + std::to_string(impl_->layer_index());
  }

  // This function returns the source code of two OpenCL kernels which calculate
  // the weight update (via gradient descent) and the backpropagated weights for
  // the next layer backwards.
  std::string GenerateTrainingKernels();

  std::string InputGradientKernelName() const {
    return "input_delta_"  + std::to_string(impl_->layer_index());
  }

  std::string WeightGradientKernelName() const {
    return "weight_delta_" + std::to_string(impl_->layer_index());
  }

  Matrix<symbolic::Expression> InputExpression() const;
  Matrix<symbolic::Expression> OutputExpression() const;

 private:
  SymbolGenerator generator_;
  std::unique_ptr<LayerImpl> impl_;
  symbolic::Environment env_;
};

}  // namespace nnet
#endif /* LAYER_H */

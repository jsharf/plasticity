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
  using Dimensions = LayerImpl::Dimensions;
  using ActivationFunctionType = LayerImpl::ActivationFunctionType;
  using VolumeDimensions = ConvolutionLayer::VolumeDimensions;
  using AreaDimensions = MaxPoolLayer::AreaDimensions;
  using FilterParams = ConvolutionLayer::FilterParams;

  // Constructors.
  Layer() {}
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
      const ActivationFunctionType& activation_function,
      SymbolGenerator* generator);

  static Layer MakeFeedForwardLayer(size_t layer_index,
                                    const Dimensions& dimensions,
                                    SymbolGenerator* generator);

  // Convolutional Layer constructors.
  static Layer MakeConvolutionLayer(size_t layer_index,
                                    const VolumeDimensions& dimensions,
                                    const FilterParams& params,
                                    SymbolGenerator* generator);

  static Layer MakeActivationLayer(
      size_t layer_index, size_t size,
      const ActivationFunctionType& activation_function,
      SymbolGenerator* generator);

  static Layer MakeSoftmaxLayer(size_t layer_index, size_t size,
                                SymbolGenerator* generator);

  static Layer MakeMaxPoolLayer(size_t layer_index,
                                const VolumeDimensions& input,
                                const AreaDimensions& output,
                                SymbolGenerator* generator);

  WeightArray weights() const;
  Matrix<symbolic::Expression> GenerateExpression(
      const Matrix<symbolic::Expression>& input);
  stats::Normal XavierInitializer();
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

  // Tread carefully... If you accidentally assign the wrong symbol generator to
  // a layer, you can end up in really weird hard to debug states.
  void SetSymbolGenerator(SymbolGenerator* generator) {
    impl_->SetSymbolGenerator(generator);
  }

  SymbolGenerator* symbol_generator() const {
    return impl_->symbol_generator();
  }

 private:
  Matrix<symbolic::Expression> InputExpression() const;

  std::unique_ptr<LayerImpl> impl_;
};

}  // namespace nnet
#endif /* LAYER_H */

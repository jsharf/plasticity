#ifndef LAYER_H
#define LAYER_H
#include "math/geometry/dynamic_matrix.h"
#include "math/memory/cl_buffer.h"
#include "math/nnet/activation_layer.h"
#include "math/nnet/convolution_layer.h"
#include "math/nnet/dense_layer.h"
#include "math/nnet/layer_dimensions.h"
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

class Nnet;

// Holds a pointer to a Layer and manages the resources.
class Layer {
 public:
  // Public exported types
  using ActivationFunctionType = LayerImpl::ActivationFunctionType;

  // Constructors.
  Layer() = delete;
  explicit Layer(std::unique_ptr<LayerImpl> &&root);
  Layer(Layer &&rhs);
  Layer(const Layer &rhs);

  // Destructor.
  virtual ~Layer() {}

  void RegisterToNetwork(nnet::Nnet *network);

  // Assignment Operators.
  Layer &operator=(const Layer &rhs) = delete;
  Layer &operator=(Layer &&rhs) = delete;

  // Dense Layer constructor. Dense layers alone do not contain an activation
  // function. This is done via a separate activation layer.
  static Layer MakeDenseLayer(size_t layer_index, const Dimensions &dimensions);

  // Convolutional Layer constructors.
  static Layer MakeConvolutionLayer(size_t layer_index,
                                    const VolumeDimensions &dimensions,
                                    const FilterParams &params);

  static Layer MakeActivationLayer(
      size_t layer_index, size_t size,
      const ActivationFunctionType &activation_function);

  static Layer MakeSoftmaxLayer(size_t layer_index, size_t size);

  static Layer MakeMaxPoolLayer(size_t layer_index,
                                const VolumeDimensions &input,
                                const AreaDimensions &output);

  std::string WeightsToString();

  stats::Normal XavierInitializer() const;
  void XavierInitializeWeights();
  void InitializeWeights(double value);

  double &W(size_t index) {
    if (index >= weights_.size()) {
      std::cerr << "Too large weight index: " << index << std::endl;
      std::exit(1);
    }
    weights_.MoveToCpu();
    return weights_[index];
  }

  memory::ClBuffer &weight_buffer() { return weights_; }

  Dimensions GetDimensions() const { return impl_->GetDimensions(); }

  // This function returns the source code of an OpenCL kernel which evaluates
  // the output of this layer, given the input.
  std::string GenerateEvaluationKernel() const;

  std::string EvaluateKernelName() const {
    return "evaluate_" + std::to_string(impl_->layer_index());
  }

  // This function returns the source code of two OpenCL kernels which calculate
  // the weight update (via gradient descent) and the backpropagated weights for
  // the next layer backwards.
  std::string GenerateTrainingKernels() const;

  std::string LayerSuffix() const {
    return impl_->layer_type() + "_" + std::to_string(impl_->layer_index());
  }

  std::string InputGradientKernelName() const {
    return "input_delta_" + LayerSuffix();
  }

  std::string WeightGradientKernelName() const {
    return "weight_delta_" + LayerSuffix();
  }

  Matrix<symbolic::Expression> InputExpression() const;
  Matrix<symbolic::Expression> OutputExpression() const;

  size_t eval_workgroup_size() const { return eval_workgroup_size_; }
  size_t weight_train_workgroup_size() const {
    return weight_train_workgroup_size_;
  }
  size_t bp_train_workgroup_size() const { return bp_train_workgroup_size_; }

 private:
  Nnet *nnet_;
  SymbolGenerator generator_;
  std::unique_ptr<LayerImpl> impl_;
  const size_t eval_workgroup_size_;
  const size_t weight_train_workgroup_size_;
  const size_t bp_train_workgroup_size_;

  // Weights are cached in the GPU between training runs.
  memory::ClBuffer weights_;
};

}  // namespace nnet
#endif /* LAYER_H */

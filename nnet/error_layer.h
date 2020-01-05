#ifndef ERROR_LAYER_H
#define ERROR_LAYER_H

#include "plasticity/geometry/dynamic_matrix.h"
#include "plasticity/nnet/layer_dimensions.h"
#include "plasticity/symbolic/expression.h"
#include "plasticity/symbolic/symbolic_util.h"

namespace nnet {

class ErrorLayer {
 public:
  ErrorLayer(LossFunction loss_function, size_t size)
      : loss_function_(loss_function),
        size_(size),
        workgroup_size_(CalculateWorkgroupSize(size_)) {}

  ErrorLayer(const ErrorLayer& rhs)
      : loss_function_(rhs.loss_function_),
        size_(rhs.size_),
        workgroup_size_(rhs.workgroup_size_) {}

  // Returns the source code for two kernels. One will compare the output of the
  // forward pass (current output) with the expected output, and generate
  // absolute error values based on the provided loss function.
  //
  // The second kernel generates gradients for the beginning of the backwards
  // pass of back prop.
  std::string GenerateErrorKernels() const;

  std::string ErrorKernelName() const { return "error_value"; }

  std::string GradientKernelName() const { return "error_gradients"; }

  size_t workgroup_size() const {
    return workgroup_size_;
  }

  size_t size() const { return size_; }

 private:
  symbolic::Expression O() const;
  symbolic::Expression E() const;
  symbolic::Expression GenerateErrorComponent() const;
  symbolic::Expression GenerateMseErrorComponent() const;
  symbolic::Expression GenerateCrossEntropyErrorComponent() const;

  LossFunction loss_function_;
  size_t size_;
  size_t workgroup_size_;
};

}  // namespace nnet

#endif  // ERROR_LAYER_H

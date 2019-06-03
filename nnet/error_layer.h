#ifndef ERROR_LAYER_H
#define ERROR_LAYER_H

#include "math/geometry/dynamic_matrix.h"
#include "math/nnet/layer_dimensions.h"
#include "math/symbolic/expression.h"
#include "math/symbolic/symbolic_util.h"

namespace nnet {

class ErrorLayer {
 public:
  ErrorLayer(LossFunction loss_function, size_t size)
      : loss_function_(loss_function), size_(size) {}

  ErrorLayer(const ErrorLayer& rhs)
      : loss_function_(rhs.loss_function_), size_(rhs.size_) {}

  // Returns the source code for two kernels. One will compare the output of the
  // forward pass (current output) with the expected output, and generate
  // absolute error values based on the provided loss function.
  //
  // The second kernel generates gradients for the beginning of the backwards
  // pass of back prop.
  std::string GenerateErrorKernels() const;

  std::string ErrorKernelName() const { return "error"; }

  std::string GradientKernelName() const { return "error_gradients"; }

  size_t size() const { return size_; }

 private:
  symbolic::Expression O() const;
  symbolic::Expression E() const;
  symbolic::Expression GenerateErrorComponent() const;
  symbolic::Expression GenerateMseErrorComponent() const;
  symbolic::Expression GenerateCrossEntropyErrorComponent() const;

  LossFunction loss_function_;
  size_t size_;
};

}  // namespace nnet

#endif  // ERROR_LAYER_H

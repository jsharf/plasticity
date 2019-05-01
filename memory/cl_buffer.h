#ifndef CL_BUFFER_H
#define CL_BUFFER_H

#include <iostream>

#include "clutil/util.h"

namespace memory {

// ASSERT for opencl calls.
#define CL_CHECK(line) do { \
  cl_int res = line; \
  if (res != CL_SUCCESS) { \
    std::cerr << "Error running line: " #line << std::endl; \
    std::cerr << "Code: " << res << std::endl; \
    std::exit(1); \
  } \
} while(0);

#define CHECK_NOTNULL(x) do { \
  if ((x) == nullptr) { \
    std::cerr << "Error, unexpected nullptr: " #x << std::endl; \
    std::exit(1); \
  } \
} while(0);

// A wrapper around cl::Buffer which allows for each transfer between CPU and
// GPU. By default, initialized to in CPU state. Access operators only allowed
// after MoveToCpu is called.
class ClBuffer {
 public:
  enum Location {
    GPU = 0,
    CPU,
  };

  ClBuffer(size_t size)
      : state_(CPU), size_(size), cpu_buffer_(size), gpu_buffer_(nullptr) {};
  ClBuffer(const ClBuffer &other)
      : state_(other.state_),
        size_(other.size_),
        cpu_buffer_(other.cpu_buffer_) {
    if (other.state_ == GPU) {
      gpu_buffer_ = new cl::Buffer(*other.gpu_buffer_);
    }
  }
  ClBuffer(ClBuffer &&other)
      : state_(other.state_),
        size_(other.size_),
        cpu_buffer_(std::move(other.cpu_buffer_)) {
    if (other.state_ == GPU) {
      gpu_buffer_ = new cl::Buffer(*other.gpu_buffer_);
      delete other.gpu_buffer_;
      other.gpu_buffer_ = nullptr;
    }
  }

  virtual ~ClBuffer() {
    if (state_ == GPU) {
      delete gpu_buffer_;
      gpu_buffer_ = nullptr;
    }
  }

  void MoveToCpu(cl::CommandQueue *cq);
  void MoveToGpu(cl::Context* context, cl::CommandQueue *cq);
  Location GetBufferLocation() { return state_; }
  size_t size() const { return size_; }

  // Only use these after MoveToCpu!
  double& operator[](size_t index);
  double operator[](size_t index) const;

  // Only use this after MoveToGpu!
  cl::Buffer *gpu_buffer() { return gpu_buffer_; }

 private:

  Location state_;
  size_t size_;
  std::vector<double> cpu_buffer_;
  cl::Buffer *gpu_buffer_ = nullptr;
};

}  // namespace memory

#endif  // CL_BUFFER_H

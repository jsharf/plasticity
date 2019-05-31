#ifndef CL_BUFFER_H
#define CL_BUFFER_H

#include <iostream>
#include <memory>
#include <utility>

#include "clutil/util.h"
#include "math/memory/buffer.h"

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
class ClBuffer : public Buffer {
public:
  using Location = Buffer::Location;

  ClBuffer()
      : state_(Buffer::CPU), cpu_buffer_(0), cq_(nullptr), context_(nullptr) {}
  ClBuffer(size_t size)
      : state_(Buffer::CPU), cpu_buffer_(size), cq_(nullptr),
        context_(nullptr) {}
  ClBuffer(const std::vector<double> &values, cl::CommandQueue *cq,
           cl::Context *context)
      : state_(Buffer::CPU), cpu_buffer_(values), cq_(cq), context_(context) {}
  ClBuffer(size_t size, cl::CommandQueue *cq, cl::Context *context)
      : state_(Buffer::CPU), cpu_buffer_(size), cq_(cq), context_(context) {
    CHECK_NOTNULL(context_);
    CHECK_NOTNULL(cq_);
  }
  ClBuffer(cl::CommandQueue *cq, cl::Context *context)
      : state_(Buffer::CPU), cpu_buffer_(0), cq_(cq), context_(context) {
    CHECK_NOTNULL(context_);
    CHECK_NOTNULL(cq_);
  }
  ClBuffer(const ClBuffer &other)
      : state_(other.state_), cpu_buffer_(other.cpu_buffer_), cq_(other.cq_),
        context_(other.context_) {
    if (other.state_ == Buffer::GPU) {
      gpu_buffer_ = std::make_unique<cl::Buffer>(*other.gpu_buffer_);
    }
    if (other.gpu_buffer_) {
      CHECK_NOTNULL(context_);
      CHECK_NOTNULL(cq_);
    }
  }
  ClBuffer(ClBuffer &&other)
      : state_(other.state_), cpu_buffer_(std::move(other.cpu_buffer_)),
        cq_(other.cq_), context_(other.context_) {
    if (other.state_ == Buffer::GPU) {
      gpu_buffer_ = std::move(other.gpu_buffer_);
      other.state_ = Buffer::CPU;
    }
    if (other.gpu_buffer_) {
      CHECK_NOTNULL(context_);
      CHECK_NOTNULL(cq_);
    }
  }

  virtual ~ClBuffer() {
    if (state_ == Buffer::GPU) {
      gpu_buffer_.reset();
    }
  }

  void RegisterBackend(cl::CommandQueue *queue, cl::Context *context) {
    CHECK_NOTNULL(cq_ = queue);
    CHECK_NOTNULL(context_ = context);
  }

  void MoveToCpu();
  void MoveToGpu();
  Location GetBufferLocation() { return state_; }
  size_t size() const;

  // Only use these after MoveToCpu!
  double &operator[](size_t index);
  const double &operator[](size_t index) const;

  // Only use this after MoveToGpu!
  const std::unique_ptr<cl::Buffer> &gpu_buffer() {
    if (state_ == Buffer::CPU) {
      std::cerr << "Requested GPU buffer when in CPU state!" << std::endl;
      std::exit(1);
    }
    return gpu_buffer_;
  }

private:
  Location state_;
  std::vector<double> cpu_buffer_;
  std::unique_ptr<cl::Buffer> gpu_buffer_;
  cl::CommandQueue *cq_ = nullptr;
  cl::Context *context_ = nullptr;
};

} // namespace memory

#endif // CL_BUFFER_H

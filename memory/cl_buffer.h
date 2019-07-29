#ifndef CL_BUFFER_H
#define CL_BUFFER_H

#include <iostream>
#include <limits>
#include <memory>
#include <utility>

#include "clutil/util.h"
#include "math/geometry/dynamic_matrix.h"
#include "math/memory/buffer.h"

// FYI for the future, this class might be a simpler interface if context is
// inferred from the buffer and CommandQueue is just initialized from
// CommandQueue::getDefault(cl_int* error);
// cl_context context = getInfo<CL_QUEUE_CONTEXT>();

namespace memory {

#define CL_CHECK(line)                                       \
  do {                                                       \
    cl_int res = line;                                       \
    if (res != CL_SUCCESS) {                                 \
      std::cerr << "Error running line " #line << std::endl; \
      std::cerr << "Code: " << res << std::endl;             \
      std::cerr << "Line no: " << __LINE__ << std::endl;     \
      std::cerr << "File: " << __FILE__ << std::endl;     \
      std::exit(1);                                          \
    }                                                        \
  } while (0);

#define CHECK_NOTNULL(x)                                          \
  do {                                                            \
    if ((x) == nullptr) {                                         \
      std::cerr << "Error, unexpected nullptr: " #x << std::endl; \
      std::exit(1);                                               \
    }                                                             \
  } while (0);

// A wrapper around cl::Buffer which allows for each transfer between CPU and
// GPU. By default, initialized to in CPU state. Access operators only allowed
// after MoveToCpu is called.
class ClBuffer : public Buffer {
 public:
  using Location = Buffer::Location;

  // Creates a clone of this buffer. If the buffer is loaded onto the GPU, uses
  // clEnqueueCopyBuffer() to efficiently copy the buffer without leaving the
  // GPU.
  ClBuffer DeepClone() {
    if (state_ == Buffer::CPU) {
      return ClBuffer(*this);
    } else {
      cl_int buffer_init;
      auto gpu_buffer = std::make_unique<cl::Buffer>(*context_, CL_MEM_READ_WRITE,
                                                 size() * sizeof(double),
                                                 nullptr, &buffer_init);
      CL_CHECK(buffer_init);
      CL_CHECK(cq_->enqueueCopyBuffer(*gpu_buffer_, *gpu_buffer, 0, 0,
                                      size() * sizeof(double)));
      return ClBuffer(cq_, context_, std::move(gpu_buffer));
    }
  }

  ClBuffer()
      : state_(Buffer::CPU), cpu_buffer_(0), cq_(nullptr), context_(nullptr) {
      }
  ClBuffer(size_t size)
      : state_(Buffer::CPU),
        cpu_buffer_(size),
        cq_(nullptr),
        context_(nullptr) {
        }
  ClBuffer(const std::vector<double> &values)
      : state_(Buffer::CPU),
        cpu_buffer_(values),
        cq_(nullptr),
        context_(nullptr) {
        }
  ClBuffer(const std::vector<double> &values, cl::CommandQueue *cq,
           cl::Context *context)
      : state_(Buffer::CPU), cpu_buffer_(values), cq_(cq), context_(context) {
  }
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
  ClBuffer(cl::CommandQueue *cq, cl::Context *context, std::unique_ptr<cl::Buffer>&& gpu_buffer)
      : state_(Buffer::GPU), cpu_buffer_(0), gpu_buffer_(std::move(gpu_buffer)), cq_(cq), context_(context) {
    CHECK_NOTNULL(context_);
    CHECK_NOTNULL(cq_);
  }
  ClBuffer(const ClBuffer &other)
      : state_(other.state_),
        cpu_buffer_(other.cpu_buffer_),
        cq_(other.cq_),
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
      : state_(other.state_),
        cpu_buffer_(std::move(other.cpu_buffer_)),
        cq_(other.cq_),
        context_(other.context_) {
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

  static std::unique_ptr<ClBuffer> MakeBufferFromColumnVector(
      Matrix<double> column_vector);

  void RegisterClBackend(cl::CommandQueue *queue, cl::Context *context) {
    MoveToCpu();
    CHECK_NOTNULL(cq_ = queue);
    CHECK_NOTNULL(context_ = context);
    MoveToGpu();
  }

  void MoveToCpu();
  void MoveToGpu();
  Location GetBufferLocation() { return state_; }
  size_t size() const;
  void resize(
      size_t new_size,
      double default_value = std::numeric_limits<double>::quiet_NaN()) override;

  // Only use these after MoveToCpu!
  double &operator[](size_t index);
  double &at(size_t index) { return (*this)[index]; }
  const double &operator[](size_t index) const;
  const double &at(size_t index) const { return (*this)[index]; }
  std::string to_string() const;

  // Only use this after MoveToGpu!
  const std::unique_ptr<cl::Buffer> &gpu_buffer() const {
    if (state_ == Buffer::CPU) {
      std::cerr << "Requested GPU buffer when in CPU state!" << std::endl;
      std::exit(1);
    }
    return gpu_buffer_;
  }

  const memory::ClBuffer &operator=(const memory::ClBuffer &rhs) {
    state_ = rhs.state_;
    cpu_buffer_ = rhs.cpu_buffer_;
    cq_ = rhs.cq_;
    context_ = rhs.context_;

    CHECK_NOTNULL(cq_);
    CHECK_NOTNULL(context_);
    if (state_ == Buffer::GPU) {
      gpu_buffer_ = std::make_unique<cl::Buffer>(*rhs.gpu_buffer());
    }

    return *this;
  }

 private:
  Location state_;
  std::vector<double> cpu_buffer_;
  std::unique_ptr<cl::Buffer> gpu_buffer_;
  cl::CommandQueue *cq_ = nullptr;
  cl::Context *context_ = nullptr;
};

}  // namespace memory

#endif  // CL_BUFFER_H

#include "math/memory/cl_buffer.h"

namespace memory {

void ClBuffer::MoveToCpu(cl::CommandQueue *cq) {
  if (state_ == CPU) { return; }
  CHECK_NOTNULL(cq);
  CHECK_NOTNULL(gpu_buffer_);
  cpu_buffer_.resize(size_);
  CL_CHECK(cq->enqueueReadBuffer(*gpu_buffer_, CL_TRUE, 0,
                                 sizeof(double) * size_, &cpu_buffer_[0]));
  delete gpu_buffer_;
  gpu_buffer_ = nullptr;
  state_ = CPU;
}

void ClBuffer::MoveToGpu(cl::Context* context, cl::CommandQueue *cq) {
  if (state_ == GPU) { return; }
  CHECK_NOTNULL(context);
  CHECK_NOTNULL(cq);
  if (gpu_buffer_ != nullptr) {
    delete gpu_buffer_;
  }
  state_ = GPU;
  cl_int buffer_init;
  if (size_ == 0) {
    // Special case. If this buffer is allocated to size zero, then allocate a
    // 1-size dummy buffer, since empty buffers aren't allowed. Don't need to
    // even intialize it.
    gpu_buffer_ = new cl::Buffer(*context, CL_MEM_READ_WRITE,
                                 sizeof(double) * 1, nullptr, &buffer_init);
    CL_CHECK(buffer_init);
    return;
  }
  gpu_buffer_ = new cl::Buffer(*context, CL_MEM_READ_WRITE, sizeof(double) * size_, nullptr, &buffer_init);
  CL_CHECK(buffer_init);
  CL_CHECK(cq->enqueueWriteBuffer(*gpu_buffer_, CL_TRUE, 0,
                                  sizeof(double) * size_, &cpu_buffer_[0]));
}

double& ClBuffer::operator[](size_t index) {
  if (state_ == GPU) {
    std::cerr << "Error: [] used while buffer is in GPU." << std::endl;
    std::exit(1);
  }

  return cpu_buffer_[index];
}

double ClBuffer::operator[](size_t index) const {
  if (state_ == GPU) {
    std::cerr << "Error: [] used while buffer is in GPU." << std::endl;
    std::exit(1);
  }

  return cpu_buffer_[index];
}

}  // namespace memory

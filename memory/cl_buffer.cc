#include "math/memory/cl_buffer.h"

namespace memory {

void ClBuffer::MoveToCpu() {
  if (state_ == CPU) { return; }
  if (!gpu_buffer_) {
    std::cerr << "Error, unexpected nullptr gpu_buffer_" << std::endl;
    std::exit(1);
  }
  if (size_ != 0) {
  cpu_buffer_.resize(size_);
  CL_CHECK(cq_->enqueueReadBuffer(*gpu_buffer_, CL_TRUE, 0,
                                  sizeof(double) * size_, &cpu_buffer_[0]));
  }
  gpu_buffer_ = nullptr;
  state_ = CPU;
}

void ClBuffer::MoveToGpu() {
  if (state_ == GPU) { return; }
  if (gpu_buffer_) {
    gpu_buffer_.reset();
  }
  state_ = GPU;
  cl_int buffer_init;
  if (size_ == 0) {
    // Special case. If this buffer is allocated to size zero, then allocate a
    // 1-size dummy buffer, since empty buffers aren't allowed. Don't need to
    // even intialize it.
    gpu_buffer_ =
        std::make_unique<cl::Buffer>(*context_, CL_MEM_READ_WRITE,
                                     sizeof(double) * 1, nullptr, &buffer_init);
    CL_CHECK(buffer_init);
    return;
  }
  gpu_buffer_ = std::make_unique<cl::Buffer>(*context_, CL_MEM_READ_WRITE,
                                             sizeof(double) * size_, nullptr,
                                             &buffer_init);
  CL_CHECK(buffer_init);
  CL_CHECK(cq_->enqueueWriteBuffer(*gpu_buffer_, CL_TRUE, 0,
                                   sizeof(double) * size_, &cpu_buffer_[0]));
}

double& ClBuffer::operator[](size_t index) {
  if (state_ == GPU) {
    std::cerr << "Error: [] used while buffer is in GPU." << std::endl;
    std::exit(1);
  }

  return cpu_buffer_[index];
}

const double& ClBuffer::operator[](size_t index) const {
  if (state_ == GPU) {
    std::cerr << "Error: [] used while buffer is in GPU." << std::endl;
    std::exit(1);
  }

  return cpu_buffer_[index];
}

}  // namespace memory

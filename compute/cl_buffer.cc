#include "math/compute/cl_buffer.h"

namespace compute {

std::unique_ptr<ClBuffer> ClBuffer::MakeBufferFromColumnVector(
    Matrix<double> column_vector) {
  std::vector<double> values = {};
  for (size_t i = 0; i < column_vector.dimensions().rows; ++i) {
    values.push_back(column_vector.at(i, 0));
  }
  return std::make_unique<ClBuffer>(values);
}

void ClBuffer::resize(size_t new_size, double default_value) {
  // This is going to kill GPU performance when resizing GPU buffers, but it's
  // not a super realistic situation and when it comes up we can optimize for it
  // quite easily.
  if (new_size == size()) {
    // Fast return optimization when nothing needs to be done. Avoids costly
    // GPU->CPU transition in some cases.
    return;
  }
  auto old_state = state_;
  if (state_ == GPU) {
    MoveToCpu();
  }
  cpu_buffer_.resize(new_size, default_value);
  if (old_state == GPU) {
    MoveToGpu();
  }
}

size_t ClBuffer::size() const {
  if (state_ == CPU) {
    return cpu_buffer_.size();
  } else {
    size_t gpu_size = 0;
    CL_CHECK(gpu_buffer_->getInfo(CL_MEM_SIZE, &gpu_size));
    // Special case.
    if (gpu_size == 1) {
      return 0;
    }
    if (gpu_size % sizeof(double) != 0) {
      std::cerr << "GPU Buffer is not an even multiple of doubles. Invalid GPU "
                   "buffer size."
                << std::endl;
      std::exit(1);
    }
    return gpu_size / sizeof(double);
  }
}

void ClBuffer::MoveToCpu(const std::unique_ptr<cl::CommandQueue>& cq) {
  cl::CommandQueue& queue = (cq) ? *cq : *cq_;
  if (state_ == CPU) {
    return;
  }
  if (!gpu_buffer_) {
    std::cerr << "Error, unexpected nullptr gpu_buffer_" << std::endl;
    std::exit(1);
  }
  if (size() != 0) {
    cpu_buffer_.resize(size());
    CL_CHECK(queue.enqueueReadBuffer(*gpu_buffer_, CL_TRUE, 0,
                                     sizeof(double) * size(), &cpu_buffer_[0]));
  }
  gpu_buffer_.reset(nullptr);
  state_ = CPU;
}

void ClBuffer::MoveToGpu(const std::unique_ptr<cl::CommandQueue>& cq) {
  cl::CommandQueue& queue = (cq) ? *cq : *cq_;
  if (state_ == GPU) {
    return;
  }
  CHECK_NOTNULL(context_);
  if (gpu_buffer_) {
    gpu_buffer_.reset();
  }
  cl_int buffer_init;
  if (size() == 0) {
    // Special case. If this buffer is allocated to size zero, then allocate a
    // 1-size dummy buffer, since empty buffers aren't allowed. Don't need to
    // even initialize it.
    gpu_buffer_ = std::make_unique<cl::Buffer>(*context_, CL_MEM_READ_WRITE, 1,
                                               nullptr, &buffer_init);
    CL_CHECK(buffer_init);
    state_ = GPU;
    return;
  }
  gpu_buffer_ = std::make_unique<cl::Buffer>(*context_, CL_MEM_READ_WRITE,
                                             sizeof(double) * size(), nullptr,
                                             &buffer_init);
  CL_CHECK(buffer_init);
  CL_CHECK(queue.enqueueWriteBuffer(*gpu_buffer_, CL_TRUE, 0,
                                    sizeof(double) * size(), &cpu_buffer_[0]));
  state_ = GPU;
}

std::string ClBuffer::to_string() const {
  if (state_ == GPU) {
    std::cerr << "Error: to_string() used while buffer is in GPU." << std::endl;
    std::exit(1);
  }

  std::stringstream out;
  out << "{\n";

  for (size_t i = 0; i < size(); ++i) {
    out << (*this)[i] << ",\n";
  }
  return out.str();
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

}  // namespace compute

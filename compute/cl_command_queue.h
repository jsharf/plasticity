#include "math/compute/command_queue.h"

#include <iostream>

namespace compute {

class ClCommandQueue : CommandQueue {
 public:
  ClCommandQueue(const ClCommandQueue& other) : queue_(other.queue_) {}
  ClCommandQueue(ClCommandQueue&& other) : queue_(std::move(other.queue_)) {}
  ClCommandQueue(const cl::Context& context, const cl::Device& device)
      : queue_(context, device) {}
  // Returns an ID to use with this program in the future.
  int AddProgram(cl::Program program);
  // Get program_id from AddProgram().
  void enqueueKernel(const std::string& kernel_name, int program_id,
                     int global_size, int workgroup_size,
                     std::vector<compute::ClBuffer> arguments) override;
  void finish() override;

 private:
  cl::CommandQueue queue_;
  std::vector<cl::Program> programs_
};

}  // namspace compute

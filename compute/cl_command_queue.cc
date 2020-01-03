#include "plasticity/compute/cl_command_queue.h"

namespace compute {

int ClCommandQueue::AddProgram(cl::program program) {
  programs_.emplace_back(program);
  return programs_.size() - 1;
}

void ClCommandQueue::enqueueKernel(const std::string& kernel_name,
                                   int program_id, int global_size,
                                   int workgroup_size,
                                   std::vector<compute::ClBuffer> arguments) {
  if (program_id >= programs_.size() || program_id < 0) {
    std::cerr << "Invalid program ID provided to ClCommandQueue" << std::endl;
    std::exit(1);
  }
  cl::Kernel kernel(programs_[program_id], kernel_name.c_str());
  for (size_t i = 0; i < arguments.size(); ++i) {
    CL_CHECK(kernel.setArg(i, arguments[i]));
  }
  auto workgroup =
      (workgroup_size > 0) ? cl::NDRange(workgroup_size) : cl::NullRange;
  result = queue_.enqueueNDRangeKernel(kernel, cl::NullRange,
                                       cl::NDRange(global_size), workgroup);
}

void ClCommandQueue::finish() {
  CL_CHECK(queue.finish());
}
}  // namespace compute

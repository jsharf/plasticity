#include "plasticity/nnet/layer_dimensions.h"

#include <algorithm>
#include <iostream>

namespace nnet {

namespace {
constexpr size_t kMaxWorkgroupSize = 1024;
}  // namespace

size_t CalculateWorkgroupSize(size_t global_num_tasks) {
  // This function takes the number of tasks to be executed in parallel and
  // determines the optimal workgroup size for opencl. I'm implementing this
  // for the GTX 1080 which has sm_61 architecture SMs (streaming
  // multiprocessor). Refer to Nvidia's OpenCL Programming guide for more
  // information.
  //
  // In short, each sm_61 SM has 128 streaming processors (cuda cores). So
  // this routine tries (in this order):
  // (1) If the number of tasks is less than 128, return global_num_tasks.
  // (2) If the number can be factored into multiples of 128, 64, or 32.
  // (3) If the number has a factor greater than 32, return that.
  // (4) Return 1;
  if (global_num_tasks <= 128) {
    return std::max(global_num_tasks, 1UL);
  }

#define RETURN_IF_DIVIDES_EVENLY(A, B)   \
  do {                                   \
    size_t _a = (A);                     \
    size_t _b = (B);                     \
    if ((_b != 0) && ((_a % _b) == 0)) { \
      return _b;                         \
    }                                    \
  } while (0);

  size_t max_workgroup_size = std::min(global_num_tasks / 2, kMaxWorkgroupSize);
  for (size_t potential_workgroup_size = 32; potential_workgroup_size < max_workgroup_size; potential_workgroup_size++) {
    if ((global_num_tasks % potential_workgroup_size) == 0) {
      return potential_workgroup_size;
    }
  }

  RETURN_IF_DIVIDES_EVENLY(global_num_tasks, 96);
  RETURN_IF_DIVIDES_EVENLY(global_num_tasks, 128);
  RETURN_IF_DIVIDES_EVENLY(global_num_tasks, 160);
  RETURN_IF_DIVIDES_EVENLY(global_num_tasks, 100);
  RETURN_IF_DIVIDES_EVENLY(global_num_tasks, 1024);
  RETURN_IF_DIVIDES_EVENLY(global_num_tasks, 512);
  RETURN_IF_DIVIDES_EVENLY(global_num_tasks, 256);

  return 1;
}

}  // namespace nnet

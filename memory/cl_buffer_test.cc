#define CATCH_CONFIG_MAIN
#include "math/third_party/catch.h"

#include <cstdlib>

#include <iostream>
#include <memory>
#include <set>
#include <tuple>
#include <unordered_map>

#include "clutil/util.h"
#include "math/memory/cl_buffer.h"

namespace memory {

// OpenCL state variables.
struct OpenClState {
  bool compiled = false;
  std::tuple<cl::Context, cl::Program> compilation_units;
  cl::Device device;
  cl::CommandQueue queue;
  std::unordered_map<std::string, cl::Kernel> kernels;
};

cl::Device SelectDevice() {
  // Select the default OpenCL device.
  cl::Platform platform = clutil::GetDefaultPlatform();
  std::vector<cl::Device> devices = clutil::GetPlatformDevices(platform);
  if (devices.size() == 0) {
    std::cerr << "No OpenCL Devices on this platform." << std::endl;
    std::exit(1);
  }
  return devices[0];
}

OpenClState CompileCl(const std::vector<std::string>& kernel_source,
                      const cl::Device& device) {
  OpenClState cl_state;
  cl_state.device = device;
  cl_state.compiled = true;
  cl_state.compilation_units = clutil::Compile(device, kernel_source);
  cl_state.queue =
      cl::CommandQueue(std::get<0>(cl_state.compilation_units), device);
  return cl_state;
}

TEST_CASE("CPU-only test", "[cpu]") {
  OpenClState cl_;
  std::vector<std::string> sources = {
      "",
  };
  cl_.device = SelectDevice();
  cl_ = CompileCl(sources, cl_.device);

  ClBuffer buf(5, &cl_.queue, &std::get<0>(cl_.compilation_units));
  for (size_t i = 0; i < 5; ++i) {
    buf[i] = i * 2 + 1;
  }
  for (size_t i = 0; i < 5; ++i) {
    REQUIRE(buf[i] == i * 2 + 1);
  }
}

TEST_CASE("Opencl tests", "[cl]") {
  OpenClState cl_;
  std::vector<std::string> sources = {
      "",
  };
  cl_.device = SelectDevice();
  cl_ = CompileCl(sources, cl_.device);

  ClBuffer buf(5, &cl_.queue, &std::get<0>(cl_.compilation_units));
  for (size_t i = 0; i < 5; ++i) {
    buf[i] = i * 2 + 1;
  }

  SECTION("Moving to opencl and back doesnt kill values!") {
    buf.MoveToGpu();
    buf.MoveToCpu();
    for (size_t i = 0; i < 5; ++i) {
      REQUIRE(buf[i] == i * 2 + 1);
    }
  }

  SECTION("Assigning via opencl doesn't kill values") {
    cl_int buffer_init;
    cl::Buffer cl_buffer(std::get<0>(cl_.compilation_units), CL_MEM_READ_WRITE,
                         5 * sizeof(double), nullptr, &buffer_init);
    if (buffer_init != CL_SUCCESS) {
      std::cerr << "Could not initialize input buffer" << std::endl;
      std::exit(1);
    }
    double inputs_buf[5];
    for (size_t i = 0; i < 5; ++i) {
      inputs_buf[i] = i * 10;
    }

    cl_int result = cl_.queue.enqueueWriteBuffer(
        cl_buffer, CL_TRUE, 0, sizeof(double) * 5, inputs_buf);
    if (result != CL_SUCCESS) {
      std::cerr << "Error enqueuing input write (Eval):  " << result
                << std::endl;
      std::exit(1);
    }

    buf.MoveToGpu();

    *buf.gpu_buffer() = cl_buffer;

    buf.MoveToCpu();

    for (size_t i = 0; i < 5; ++i) {
      REQUIRE(buf[i] == i * 10);
    }
  }

  SECTION("opencl copy constructor") {
    cl_int buffer_init;
    cl::Buffer cl_buffer(std::get<0>(cl_.compilation_units), CL_MEM_READ_WRITE,
                         5 * sizeof(double), nullptr, &buffer_init);
    if (buffer_init != CL_SUCCESS) {
      std::cerr << "Could not initialize input buffer" << std::endl;
      std::exit(1);
    }
    double inputs_buf[5];
    for (size_t i = 0; i < 5; ++i) {
      inputs_buf[i] = i * 10;
    }

    cl_int result = cl_.queue.enqueueWriteBuffer(
        cl_buffer, CL_TRUE, 0, sizeof(double) * 5, inputs_buf);
    if (result != CL_SUCCESS) {
      std::cerr << "Error enqueuing input write (Eval):  " << result
                << std::endl;
      std::exit(1);
    }

    buf.MoveToGpu();

    *buf.gpu_buffer() = cl_buffer;

    ClBuffer buf2(buf);

    buf2.MoveToCpu();

    for (size_t i = 0; i < 5; ++i) {
      REQUIRE(buf2[i] == i * 10);
      buf2[i] = i * 2 + 1;
    }

    buf2.MoveToGpu();

    buf.MoveToCpu();
    for (size_t i = 0; i < 5; ++i) {
      REQUIRE(buf[i] == i * 10);
    }
  }

  SECTION("Assigning via cpu doesn't kill opencl values") {
    cl_int buffer_init;
    cl::Buffer cl_buffer(std::get<0>(cl_.compilation_units), CL_MEM_READ_WRITE,
                         5 * sizeof(double), nullptr, &buffer_init);
    if (buffer_init != CL_SUCCESS) {
      std::cerr << "Could not initialize input buffer" << std::endl;
      std::exit(1);
    }
    double inputs_buf[5];
    for (size_t i = 0; i < 5; ++i) {
      inputs_buf[i] = i * 10;
    }

    cl_int result = cl_.queue.enqueueWriteBuffer(
        cl_buffer, CL_TRUE, 0, sizeof(double) * 5, inputs_buf);
    if (result != CL_SUCCESS) {
      std::cerr << "Error enqueuing input write (Eval):  " << result
                << std::endl;
      std::exit(1);
    }

    buf.MoveToGpu();
    buf.MoveToGpu();

    *buf.gpu_buffer() = cl_buffer;

    buf.MoveToCpu();
    buf.MoveToGpu();
    buf.MoveToCpu();

    for (size_t i = 0; i < 5; ++i) {
      buf[i] = i * 2 + 1;
    }

    buf.MoveToCpu();
    buf.MoveToGpu();
    buf.MoveToCpu();

    for (size_t i = 0; i < 5; ++i) {
      REQUIRE(buf[i] == i * 2 + 1);
    }
  }
}

}  // namespace memory

cc_library(
    name = "buffer",
    hdrs = ["buffer.h"],
    copts = [
        "--std=c++1z",
    ],
)

cc_library(
    name = "cl_buffer",
    hdrs = ["cl_buffer.h"],
    srcs = ["cl_buffer.cc"],
    copts = [
        "--std=c++1z",
    ],
    visibility = ["//math"],
    deps = [
        ":buffer",
        "//clutil:util",
        "//math/geometry:dynamic_matrix",
    ],
)

cc_binary(
    name = "cl_buffer_test",
    srcs = ["cl_buffer_test.cc"],
    copts = [
        "-std=c++1z",
    ],
    linkopts = select({
        "//clutil:osx": ["-framework OpenCL"],
        "//clutil:linux": ["-lOpenCL", "-L/usr/local/cuda-8.0/targets/x86_64-linux/lib"],
        "//conditions:default": ["-lOpenCL", "-L/usr/local/cuda-8.0/targets/x86_64-linux/lib"],
    }),
    deps = [
        ":cl_buffer",
        "//clutil:util",
        "//math/third_party:catch2",
    ],
)

cc_library(
    name = "command_queue",
    hdrs = ["command_queue.h"],
    copts = [
        "--std=c++1z",
    ],
)

cc_library(
    name = "cl_command_queue",
    hdrs = ["cl_command_queue.h"],
    srcs = ["cl_command_queue.cc"],
    copts = [
        "--std=c++1z",
    ],
    visibility = ["//math"],
    deps = [
        ":command_queue",
        "//clutil:util",
        "//math/geometry:dynamic_matrix",
    ],
)

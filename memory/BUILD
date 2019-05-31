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

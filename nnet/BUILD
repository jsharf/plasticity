cc_library(
    name = "architecture",
    hdrs = ["architecture.h"],
    copts = [
        "--std=c++1z",
    ],
    visibility = ["//math"],
    deps = [
        ":layer",
        ":layer_dimensions",
        "//math/symbolic",
        "//math/symbolic:symbolic_util",
    ],
)

cc_library(
    name = "nnet",
    hdrs = ["nnet.h"],
    copts = [
        "--std=c++1z",
    ],
    data = glob(["kernels/*"]),
    linkopts = [
        "-pthread",
    ],
    visibility = ["//math"],
    deps = [
        ":architecture",
        ":layer",
        ":layer_dimensions",
        "//clutil:util",
        "//math/geometry:dynamic_matrix",
        "//math/memory:cl_buffer",
        "//math/stats:normal",
        "//math/symbolic",
        "//math/symbolic:symbolic_util",
    ],
)

cc_library(
    name = "layer",
    srcs = ["layer.cc"],
    hdrs = ["layer.h"],
    copts = [
        "--std=c++1z",
    ],
    linkopts = [
        "-pthread",
    ],
    visibility = ["//math"],
    deps = [
        ":activation_layer",
        ":convolution_layer",
        ":dense_layer",
        ":layer_impl",
        ":max_pool_layer",
        ":softmax_layer",
        "//clutil:util",
        "//math/codegen:codegen",
        "//math/geometry:dynamic_matrix",
        "//math/memory:cl_buffer",
        "//math/stats:normal",
        "//math/symbolic",
        "//math/symbolic:symbolic_util",
    ],
)

cc_library(
    name = "layer_impl",
    srcs = ["layer_impl.cc"],
    hdrs = ["layer_impl.h"],
    copts = [
        "--std=c++1z",
    ],
    visibility = ["//math"],
    deps = [
        ":layer_dimensions",
        "//math/codegen:codegen",
        "//math/geometry:dynamic_matrix",
        "//math/stats:normal",
        "//math/symbolic",
        "//math/symbolic:symbolic_util",
    ],
)

cc_library(
    name = "symbol_generator",
    hdrs = ["symbol_generator.h"],
    srcs = ["symbol_generator.cc"],
    copts = [
        "--std=c++1z",
    ],
    visibility = ["//math"],
    deps = [
        ":layer_dimensions",
        ":layer_impl",
        "//math/codegen:codegen",
        "//math/codegen:codegen_util",
        "//math/geometry:dynamic_matrix",
        "//math/stats:normal",
        "//math/symbolic",
        "//math/symbolic:symbolic_util",
    ],
)

cc_library(
    name = "layer_dimensions",
    hdrs = ["layer_dimensions.h"],
    copts = [
        "--std=c++1z",
    ],
    visibility = ["//math"],
    deps = [
    ],
)

cc_library(
    name = "dense_layer",
    srcs = ["dense_layer.cc"],
    hdrs = ["dense_layer.h"],
    copts = [
        "--std=c++1z",
    ],
    visibility = ["//math"],
    deps = [
        ":layer_dimensions",
        ":layer_impl",
        ":symbol_generator",
        "//math/codegen:codegen",
        "//math/geometry:dynamic_matrix",
        "//math/stats:normal",
        "//math/symbolic",
        "//math/symbolic:symbolic_util",
    ],
)

cc_library(
    name = "activation_layer",
    srcs = ["activation_layer.cc"],
    hdrs = ["activation_layer.h"],
    copts = [
        "--std=c++1z",
    ],
    visibility = ["//math"],
    deps = [
        ":layer_dimensions",
        ":layer_impl",
        ":symbol_generator",
        "//math/codegen:codegen",
        "//math/geometry:dynamic_matrix",
        "//math/stats:normal",
        "//math/symbolic",
    ],
)

cc_library(
    name = "softmax_layer",
    srcs = ["softmax_layer.cc"],
    hdrs = ["softmax_layer.h"],
    copts = [
        "--std=c++1z",
    ],
    visibility = ["//math"],
    deps = [
        ":layer_dimensions",
        ":layer_impl",
        ":symbol_generator",
        "//math/codegen:codegen",
        "//math/geometry:dynamic_matrix",
        "//math/stats:normal",
        "//math/symbolic",
        "//math/symbolic:symbolic_util",
    ],
)

cc_library(
    name = "convolution_layer",
    srcs = ["convolution_layer.cc"],
    hdrs = ["convolution_layer.h"],
    copts = [
        "--std=c++1z",
    ],
    visibility = ["//math"],
    deps = [
        ":layer_dimensions",
        ":layer_impl",
        ":symbol_generator",
        "//math/codegen:codegen",
        "//math/geometry:dynamic_matrix",
        "//math/stats:normal",
        "//math/symbolic",
        "//math/symbolic:symbolic_util",
    ],
)

cc_library(
    name = "max_pool_layer",
    srcs = ["max_pool_layer.cc"],
    hdrs = ["max_pool_layer.h"],
    copts = [
        "--std=c++1z",
    ],
    visibility = ["//math"],
    deps = [
        ":convolution_layer",
        ":layer_dimensions",
        ":layer_impl",
        "//math/codegen:codegen",
        "//math/geometry:dynamic_matrix",
        "//math/stats:normal",
        "//math/symbolic",
        "//math/symbolic:symbolic_util",
    ],
)

cc_binary(
    name = "nnet_demo",
    srcs = ["nnet_demo.cc"],
    copts = [
        "--std=c++1z",
    ],
    linkopts = select({
        "//clutil:osx": ["-framework OpenCL"],
        "//clutil:linux": ["-lOpenCL", "-L/usr/local/cuda-8.0/targets/x86_64-linux/lib"],
        "//conditions:default": ["-lOpenCL", "-L/usr/local/cuda-8.0/targets/x86_64-linux/lib"],
    }),
    visibility = ["//math"],
    deps = [
        ":nnet",
        "//math/geometry:dynamic_matrix",
        "//math/symbolic",
    ],
)

cc_binary(
    name = "circle_test",
    srcs = ["circle_test.cc"],
    copts = [
        "--std=c++1z",
    ],
    linkopts = select({
        "//clutil:osx": ["-framework OpenCL"],
        "//clutil:linux": ["-lOpenCL", "-L/usr/local/cuda-8.0/targets/x86_64-linux/lib"],
        "//conditions:default": ["-lOpenCL", "-L/usr/local/cuda-8.0/targets/x86_64-linux/lib"],
    }),
    visibility = ["//math"],
    deps = [
        ":layer",
        ":nnet",
        "//math/geometry:dynamic_matrix",
        "//math/symbolic",
    ],
)

cc_binary(
    name = "cifar_test",
    srcs = ["cifar_test.cc"],
    copts = [
        "--std=c++1z",
    ],
    data = [
        "//math/nnet/data:cifar-10",
    ],
    linkopts = select({
        "//clutil:osx": ["-framework OpenCL"],
        "//clutil:linux": ["-lOpenCL", "-L/usr/local/cuda-8.0/targets/x86_64-linux/lib"],
        "//conditions:default": ["-lOpenCL", "-L/usr/local/cuda-8.0/targets/x86_64-linux/lib"],
    }),
    visibility = ["//math"],
    deps = [
        ":convolution_layer",
        ":layer",
        ":nnet",
        "//math/geometry:dynamic_matrix",
        "//math/symbolic",
    ],
)

cc_test(
    name = "nnet_test",
    srcs = ["nnet_test.cc"],
    copts = [
        "--std=c++1z",
    ],
    linkopts = select({
        "//clutil:osx": ["-framework OpenCL"],
        "//clutil:linux": ["-lOpenCL", "-L/usr/local/cuda-8.0/targets/x86_64-linux/lib"],
        "//conditions:default": ["-lOpenCL", "-L/usr/local/cuda-8.0/targets/x86_64-linux/lib"],
    }),
    deps = [
        ":layer",
        ":nnet",
        ":symbol_generator",
        "//math/geometry:dynamic_matrix",
        "//math/stats:normal",
        "//math/symbolic",
        "//math/symbolic:symbolic_util",
        "//math/third_party:catch2",
    ],
)

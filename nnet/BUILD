cc_library(
    name = "architecture",
    hdrs = ["architecture.h"],
    copts = [
        "--std=c++1z",
    ],
    visibility = ["//:plasticity"],
    deps = [
        ":layer",
        ":layer_dimensions",
        "//symbolic",
        "//symbolic:symbolic_util",
    ],
)

filegroup(
    name = "kernels",
    srcs = glob(["kernels/*"]),
)

cc_library(
    name = "nnet",
    hdrs = ["nnet.h"],
    copts = [
        "--std=c++1z",
    ],
    data = [":kernels"],
    linkopts = [
        "-pthread",
    ],
    visibility = ["//:plasticity"],
    deps = [
        ":architecture",
        ":error_layer",
        ":layer",
        ":layer_dimensions",
        "@clutil//:util",
        "//geometry:dynamic_matrix",
        "//compute:cl_buffer",
        "//stats:normal",
        "//symbolic",
        "//symbolic:symbolic_util",
    ],
)

cc_library(
    name = "layer",
    srcs = ["layer.cc"],
    hdrs = [
        "architecture.h",
        "layer.h",
        "nnet.h",
    ],
    copts = [
        "--std=c++1z",
    ],
    linkopts = [
        "-pthread",
    ],
    visibility = ["//:plasticity"],
    deps = [
        ":activation_layer",
        ":convolution_layer",
        ":dense_layer",
        ":error_layer",
        ":layer_impl",
        ":max_pool_layer",
        ":softmax_layer",
        "@clutil//:util",
        "//codegen",
        "//geometry:dynamic_matrix",
        "//compute:cl_buffer",
        "//stats:normal",
        "//symbolic",
        "//symbolic:symbolic_util",
    ],
)

cc_library(
    name = "error_layer",
    srcs = ["error_layer.cc"],
    hdrs = ["error_layer.h"],
    copts = [
        "--std=c++1z",
    ],
    visibility = ["//:plasticity"],
    deps = [
        ":layer_dimensions",
        "@clutil//:util",
        "//geometry:dynamic_matrix",
        "//symbolic",
        "//symbolic:symbolic_util",
    ],
)

cc_library(
    name = "layer_impl",
    srcs = ["layer_impl.cc"],
    hdrs = ["layer_impl.h"],
    copts = [
        "--std=c++1z",
    ],
    visibility = ["//:plasticity"],
    deps = [
        ":layer_dimensions",
        "//codegen",
        "//geometry:dynamic_matrix",
        "//stats:normal",
        "//symbolic",
        "//symbolic:symbolic_util",
    ],
)

cc_library(
    name = "symbol_generator",
    srcs = ["symbol_generator.cc"],
    hdrs = ["symbol_generator.h"],
    copts = [
        "--std=c++1z",
    ],
    visibility = ["//:plasticity"],
    deps = [
        ":layer_dimensions",
        ":layer_impl",
        "//codegen",
        "//codegen:codegen_util",
        "//geometry:dynamic_matrix",
        "//stats:normal",
        "//symbolic",
        "//symbolic:symbolic_util",
    ],
)

cc_library(
    name = "layer_dimensions",
    hdrs = ["layer_dimensions.h"],
    srcs = ["layer_dimensions.cc"],
    copts = [
        "--std=c++1z",
    ],
    visibility = ["//:plasticity"],
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
    visibility = ["//:plasticity"],
    deps = [
        ":layer_dimensions",
        ":layer_impl",
        ":symbol_generator",
        "//codegen",
        "//geometry:dynamic_matrix",
        "//stats:normal",
        "//symbolic",
        "//symbolic:symbolic_util",
    ],
)

cc_library(
    name = "activation_layer",
    srcs = ["activation_layer.cc"],
    hdrs = ["activation_layer.h"],
    copts = [
        "--std=c++1z",
    ],
    visibility = ["//:plasticity"],
    deps = [
        ":layer_dimensions",
        ":layer_impl",
        ":symbol_generator",
        "//codegen",
        "//geometry:dynamic_matrix",
        "//stats:normal",
        "//symbolic",
    ],
)

cc_library(
    name = "softmax_layer",
    srcs = ["softmax_layer.cc"],
    hdrs = ["softmax_layer.h"],
    copts = [
        "--std=c++1z",
    ],
    visibility = ["//:plasticity"],
    deps = [
        ":layer_dimensions",
        ":layer_impl",
        ":symbol_generator",
        "//codegen",
        "//geometry:dynamic_matrix",
        "//stats:normal",
        "//symbolic",
        "//symbolic:symbolic_util",
    ],
)

cc_library(
    name = "convolution_layer",
    srcs = ["convolution_layer.cc"],
    hdrs = ["convolution_layer.h"],
    copts = [
        "--std=c++1z",
    ],
    visibility = ["//:plasticity"],
    deps = [
        ":layer_dimensions",
        ":layer_impl",
        ":symbol_generator",
        "//codegen",
        "//geometry:dynamic_matrix",
        "//stats:normal",
        "//symbolic",
        "//symbolic:symbolic_util",
    ],
)

cc_library(
    name = "max_pool_layer",
    srcs = ["max_pool_layer.cc"],
    hdrs = ["max_pool_layer.h"],
    copts = [
        "--std=c++1z",
    ],
    visibility = ["//:plasticity"],
    deps = [
        ":convolution_layer",
        ":layer_dimensions",
        ":layer_impl",
        "//codegen",
        "//geometry:dynamic_matrix",
        "//stats:normal",
        "//symbolic",
        "//symbolic:symbolic_util",
    ],
)

cc_binary(
    name = "circle_test",
    srcs = ["circle_test.cc"],
    copts = [
        "--std=c++1z",
    ],
    linkopts = select({
        "@clutil//:osx": ["-framework OpenCL"],
        "@clutil//:linux": [
            "-lOpenCL",
            "-L/usr/local/cuda-8.0/targets/x86_64-linux/lib",
            "-L/usr/lib/x86_64-linux-gnu/",
        ],
        "//conditions:default": [
            "-lOpenCL",
            "-L/usr/local/cuda-8.0/targets/x86_64-linux/lib",
            "-L/usr/lib/x86_64-linux-gnu/",
        ],
    }),
    visibility = ["//:plasticity"],
    deps = [
        ":layer",
        ":nnet",
        "//geometry:dynamic_matrix",
        "//symbolic",
    ],
)

cc_binary(
    name = "cifar_test",
    srcs = ["cifar_test.cc"],
    copts = [
        "--std=c++1z",
        "-O3",
    ],
    data = [
        "//nnet/data:cifar-10",
    ],
    linkopts = select({
        "@clutil//:osx": ["-framework OpenCL"],
        "@clutil//:linux": [
            "-lOpenCL",
            "-L/usr/local/cuda-8.0/targets/x86_64-linux/lib",
            "-L/usr/lib/x86_64-linux-gnu/",
        ],
        "//conditions:default": [
            "-lOpenCL",
            "-L/usr/local/cuda-8.0/targets/x86_64-linux/lib",
            "-L/usr/lib/x86_64-linux-gnu/",
        ],
    }),
    visibility = ["//:plasticity"],
    deps = [
        ":convolution_layer",
        ":layer",
        ":nnet",
        "//geometry:dynamic_matrix",
        "//symbolic",
    ],
)

cc_binary(
    name = "cifar_test_gprof",
    srcs = ["cifar_test.cc"],
    copts = [
        "--std=c++1z",
        "-g",
        "-pg",
        "-O3",
    ],
    data = [
        "//nnet/data:cifar-10",
    ],
    linkopts = select({
        "@clutil//:osx": ["-framework OpenCL"],
        "@clutil//:linux": [
            "-lOpenCL",
            "-L/usr/local/cuda-8.0/targets/x86_64-linux/lib",
            "-L/usr/lib/x86_64-linux-gnu/",
        ],
        "//conditions:default": [
            "-lOpenCL",
            "-L/usr/local/cuda-8.0/targets/x86_64-linux/lib",
            "-L/usr/lib/x86_64-linux-gnu/",
        ],
    }) + [
        "-g",
        "-pg",
    ],
    visibility = ["//:plasticity"],
    deps = [
        ":convolution_layer",
        ":layer",
        ":nnet",
        "//geometry:dynamic_matrix",
        "//symbolic",
    ],
)

cc_binary(
    name = "cifar_test_gprof_short",
    srcs = ["cifar_test.cc"],
    copts = [
        "--std=c++1z",
        "-g",
        "-pg",
        "-O3",
    ],
    data = [
        "//nnet/data:cifar-10",
    ],
    args = [
        "--short",
    ],
    linkopts = select({
        "@clutil//:osx": ["-framework OpenCL"],
        "@clutil//:linux": [
            "-lOpenCL",
            "-L/usr/local/cuda-8.0/targets/x86_64-linux/lib",
            "-L/usr/lib/x86_64-linux-gnu/",
        ],
        "//conditions:default": [
            "-lOpenCL",
            "-L/usr/local/cuda-8.0/targets/x86_64-linux/lib",
            "-L/usr/lib/x86_64-linux-gnu/",
        ],
    }) + [
        "-g",
        "-pg",
    ],
    visibility = ["//:plasticity"],
    deps = [
        ":convolution_layer",
        ":layer",
        ":nnet",
        "//geometry:dynamic_matrix",
        "//symbolic",
    ],
)

genrule(
    name = "cifar_profile",
    srcs = [
        ":kernels",
        "//nnet/data:cifar-10",
    ],
    outs = ["profile.dot", "profile.out"],
    cmd = "$(execpath :cifar_test_gprof_short); gprof $(execpath :cifar_test_gprof_short) > $(@D)/profile.out; gprof2dot $(@D)/profile.out > $(@D)/profile.dot",
    tools = [
        ":cifar_test_gprof_short",
    ],
)

cc_test(
    name = "nnet_test",
    srcs = ["nnet_test.cc"],
    copts = [
        "--std=c++1z",
    ],
    linkopts = select({
        "@clutil//:osx": ["-framework OpenCL"],
        "@clutil//:linux": [
            "-lOpenCL",
            "-lSegFault",
            "-L/usr/local/cuda-8.0/targets/x86_64-linux/lib",
            "-L/usr/lib/x86_64-linux-gnu/",
        ],
        "//conditions:default": [
            "-lOpenCL",
            "-L/usr/local/cuda-8.0/targets/x86_64-linux/lib",
            "-L/usr/lib/x86_64-linux-gnu/",
        ],
    }),
    deps = [
        ":layer",
        ":nnet",
        ":symbol_generator",
        "//geometry:dynamic_matrix",
        "//stats:normal",
        "//symbolic",
        "//symbolic:symbolic_util",
        "//third_party:catch2",
    ],
)

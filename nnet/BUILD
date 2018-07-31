cc_library(
    name = "architecture",
    visibility = ["//math"],
    hdrs = ["architecture.h"],
    deps = [
        ":layer",
        ":layer_dimensions",
        "//math/symbolic:symbolic",
        "//math/symbolic:symbolic_util",
    ],
)
    
cc_library(
    name = "nnet",
    visibility = ["//math"],
    hdrs = ["nnet.h"],
    data = glob(["kernels/*"]),
    linkopts = [
        "-pthread",
    ],
    deps = [
        ":architecture",
        ":layer",
        ":layer_dimensions",
        "//math/geometry:dynamic_matrix",
        "//math/symbolic:symbolic",
        "//math/symbolic:symbolic_util",
        "//math/stats:normal",
        "//clutil:util",
    ],
)

cc_library(
    name = "layer",
    visibility = ["//math"],
    hdrs = ["layer.h"],
    srcs = ["layer.cc"],
    copts = [
      "--std=c++1z",
    ],
    linkopts = [
      "-pthread",
    ],
    deps = [
        ":activation_layer",
        ":convolution_layer",
        ":dense_layer",
        ":layer_impl",
        ":max_pool_layer",
        ":softmax_layer",
        "//math/geometry:dynamic_matrix",
        "//math/symbolic:symbolic",
        "//math/symbolic:symbolic_util",
        "//math/stats:normal",
    ],
)

cc_library(
    name = "layer_impl",
    visibility = ["//math"],
    hdrs = ["layer_impl.h"],
    srcs = ["layer_impl.cc"],
    copts = [
      "--std=c++1z",
    ],
    deps = [
        ":layer_dimensions",
        "//math/geometry:dynamic_matrix",
        "//math/symbolic:symbolic",
        "//math/symbolic:symbolic_util",
        "//math/stats:normal",
    ],
)

cc_library(
    name = "symbol_generator",
    visibility = ["//math"],
    hdrs = ["symbol_generator.h"],
    copts = [
      "--std=c++1z",
    ],
    deps = [
        ":layer_impl",
        ":layer_dimensions",
        "//math/geometry:dynamic_matrix",
        "//math/symbolic:symbolic",
        "//math/stats:normal",
    ],
)

cc_library(
    name = "layer_dimensions",
    visibility = ["//math"],
    hdrs = ["layer_dimensions.h"],
    copts = [
      "--std=c++1z",
    ],
    deps = [
    ],
)

cc_library(
    name = "dense_layer",
    visibility = ["//math"],
    hdrs = ["dense_layer.h"],
    srcs = ["dense_layer.cc"],
    copts = [
      "--std=c++1z",
    ],
    deps = [
        ":layer_impl",
        ":symbol_generator",
        ":layer_dimensions",
        "//math/geometry:dynamic_matrix",
        "//math/symbolic:symbolic",
        "//math/symbolic:symbolic_util",
        "//math/stats:normal",
    ],
)

cc_library(
    name = "activation_layer",
    visibility = ["//math"],
    hdrs = ["activation_layer.h"],
    srcs = ["activation_layer.cc"],
    copts = [
      "--std=c++1z",
    ],
    deps = [
        ":layer_impl",
        ":layer_dimensions",
        "//math/geometry:dynamic_matrix",
        "//math/symbolic:symbolic",
        "//math/stats:normal",
    ],
)

cc_library(
    name = "softmax_layer",
    visibility = ["//math"],
    hdrs = ["softmax_layer.h"],
    srcs = ["softmax_layer.cc"],
    copts = [
      "--std=c++1z",
    ],
    deps = [
        ":layer_impl",
        ":layer_dimensions",
        "//math/geometry:dynamic_matrix",
        "//math/stats:normal",
        "//math/symbolic:symbolic",
        "//math/symbolic:symbolic_util",
    ],
)

cc_library(
    name = "convolution_layer",
    visibility = ["//math"],
    hdrs = ["convolution_layer.h"],
    srcs = ["convolution_layer.cc"],
    copts = [
      "--std=c++1z",
    ],
    deps = [
        ":layer_impl",
        ":symbol_generator",
        ":layer_dimensions",
        "//math/geometry:dynamic_matrix",
        "//math/symbolic:symbolic",
        "//math/symbolic:symbolic_util",
        "//math/stats:normal",
    ],
)

cc_library(
    name = "max_pool_layer",
    visibility = ["//math"],
    hdrs = ["max_pool_layer.h"],
    srcs = ["max_pool_layer.cc"],
    copts = [
      "--std=c++1z",
    ],
    deps = [
        ":convolution_layer",
        ":layer_impl",
        ":layer_dimensions",
        "//math/geometry:dynamic_matrix",
        "//math/symbolic:symbolic",
        "//math/symbolic:symbolic_util",
        "//math/stats:normal",
    ],
)

cc_binary(
    name = "nnet_demo",
    visibility = ["//math"],
    srcs = ["nnet_demo.cc"],
    copts = [
      "--std=c++1z",
    ],
    linkopts = select({
        "//clutil:osx" : ["-framework OpenCL"],
        "//clutil:linux" : ["-lOpenCL"],
        "//conditions:default" : ["-lOpenCL"],
    }),
    deps = [
        ":nnet",
        "//math/symbolic:symbolic",
        "//math/geometry:dynamic_matrix",
    ],
)

cc_binary(
    name = "circle_test",
    visibility = ["//math"],
    srcs = ["circle_test.cc"],
    copts = [
      "--std=c++1z",
    ],
    linkopts = select({
        "//clutil:osx" : ["-framework OpenCL"],
        "//clutil:linux" : ["-lOpenCL"],
        "//conditions:default" : ["-lOpenCL"],
    }),
    deps = [
        ":nnet",
        ":layer",
        "//math/symbolic:symbolic",
        "//math/geometry:dynamic_matrix",
    ],
)

cc_binary(
    name = "cifar_test",
    visibility = ["//math"],
    srcs = ["cifar_test.cc"],
    copts = [
      "--std=c++1z",
    ],
    linkopts = select({
        "//clutil:osx" : ["-framework OpenCL"],
        "//clutil:linux" : ["-lOpenCL"],
        "//conditions:default" : ["-lOpenCL"],
    }),
    deps = [
        ":nnet",
        ":layer",
        ":convolution_layer",
        "//math/symbolic:symbolic",
        "//math/geometry:dynamic_matrix",
    ],
    data = [
        "//math/nnet/data:cifar-10"
    ],
)

cc_test(
    name = "nnet_test",
    srcs = ["nnet_test.cc"],
    copts = [
        "--std=c++1z",
    ],
    linkopts = select({
        "//clutil:osx" : ["-framework OpenCL"],
        "//clutil:linux" : ["-lOpenCL"],
        "//conditions:default" : ["-lOpenCL"],
    }),
    deps = [
        ":nnet",
        ":layer",
        ":symbol_generator",
        "//math/symbolic:symbolic",
        "//math/symbolic:symbolic_util",
        "//math/geometry:dynamic_matrix",
        "//math/third_party:catch2",
    ],
)

cc_library(
    name = "nnet",
    visibility = ["//math"],
    hdrs = ["nnet.h"],
    deps = [
        ":layer",
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
    deps = [
        ":convolution_layer",
        ":feed_forward_layer",
        ":layer_impl",
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
    copts = [
      "--std=c++1z",
    ],
    deps = [
        "//math/geometry:dynamic_matrix",
        "//math/symbolic:symbolic",
        "//math/symbolic:symbolic_util",
        "//math/stats:normal",
    ],
)

cc_library(
    name = "feed_forward_layer",
    visibility = ["//math"],
    hdrs = ["feed_forward_layer.h"],
    srcs = ["feed_forward_layer.cc"],
    copts = [
      "--std=c++1z",
    ],
    deps = [
        ":layer_impl",
        "//math/geometry:dynamic_matrix",
        "//math/symbolic:symbolic",
        "//math/symbolic:symbolic_util",
        "//math/stats:normal",
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
        "//math/geometry:dynamic_matrix",
        "//math/symbolic:symbolic",
        "//math/symbolic:symbolic_util",
        "//math/stats:normal",
    ],
)

cc_binary(
    name = "nnet_test",
    visibility = ["//math"],
    srcs = ["nnet_test.cc"],
    copts = [
      "--std=c++1z",
    ],
    linkopts = [
        "-lOpenCL",
    ],
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
    linkopts = [
        "-lOpenCL",
    ],
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
    linkopts = [
        "-lOpenCL",
    ],
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

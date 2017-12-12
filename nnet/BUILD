cc_library(
    name = "nnet",
    visibility = ["//math"],
    hdrs = ["nnet.h"],
    deps = [
        ":layer",
        "//math/geometry:matrix",
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
    deps = [
        "//math/geometry:matrix",
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
        "//math/geometry:matrix",
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
        "//math/symbolic:symbolic",
        "//math/geometry:matrix",
    ],
)


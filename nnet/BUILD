cc_library(
    name = "nnet",
    visibility = ["//math"],
    hdrs = ["nnet.h"],
    copts = [
      "-fopenmp",
    ],
    deps = [
        "//math/geometry:matrix",
        "//math/symbolic:symbolic",
        "//math/symbolic:symbolic_util",
    ],
)


cc_binary(
    name = "nnet_test",
    visibility = ["//math"],
    srcs = ["nnet_test.cc"],
    copts = [
      "-fopenmp",
      "--std=c++1z",
    ],
    linkopts = [
      "-fopenmp",
    ],
    deps = [
        ":nnet",
        "//math/symbolic:symbolic",
        "//math/geometry:matrix",
    ],
)


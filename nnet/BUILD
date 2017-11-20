cc_library(
    name = "nnet",
    visibility = ["//math"],
    hdrs = ["nnet.h"],
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
        "--std=c++1z",
    ],
    deps = [
        ":nnet",
        "//math/symbolic:symbolic",
        "//math/geometry:matrix",
    ],
)


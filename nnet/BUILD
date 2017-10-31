cc_library(
    name = "nnet",
    visibility = ["//math"],
    hdrs = ["nnet.h"],
    srcs = ["nnet.cc"],
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
    deps = [
        ":nnet",
    ],
)


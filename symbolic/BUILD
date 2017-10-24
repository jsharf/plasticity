cc_library(
    name = "symbolic",
    visibility = ["//math"],
    hdrs = [
        "expression.h",
        "numeric_value.h",
    ],
    srcs = [
        "expression.cc",
        "numeric_value.cc",
    ],
    copts = [
        "-std=c++1z",
    ],
)

cc_binary(
    name = "symbolic_test",
    srcs = ["symbolic_test.cc"],
    deps = [":symbolic"],
    copts = [
        "-std=c++1z",
    ],
)

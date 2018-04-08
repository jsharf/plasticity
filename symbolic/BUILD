cc_library(
    name = "symbolic",
    visibility = ["//math"],
    hdrs = [
        "expression.h",
        "expression_node.h",
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

cc_library(
    name = "symbolic_util",
    visibility = ["//math"],
    hdrs = [
        "symbolic_util.h",
    ],
    srcs = ["symbolic_util.cc"],
    copts = [
        "-std=c++1z",
    ],
    deps = [
        ":symbolic",
        "//math/geometry:dynamic_matrix",
    ],
)

cc_binary(
    name = "symbolic_test",
    srcs = ["symbolic_test.cc"],
    copts = [
        "-std=c++1z",
    ],
    deps = [
        ":symbolic",
        ":symbolic_util",
    ],
)

cc_binary(
    name = "simple_test",
    srcs = ["simple_test.cc"],
    copts = [
        "-std=c++1z",
    ],
    deps = [
        ":symbolic",
        ":symbolic_util",
    ],
)


cc_library(
    name = "symbolic",
    srcs = [
        "expression.cc",
        "numeric_value.cc",
    ],
    hdrs = [
        "expression.h",
        "expression_node.h",
        "numeric_value.h",
    ],
    copts = [
        "-std=c++1z",
    ],
    visibility = ["//math"],
)

cc_library(
    name = "symbolic_util",
    srcs = ["symbolic_util.cc"],
    hdrs = [
        "symbolic_util.h",
    ],
    copts = [
        "-std=c++1z",
    ],
    visibility = ["//math"],
    deps = [
        ":symbolic",
        "//math/geometry:dynamic_matrix",
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

cc_test(
    name = "symbolic_test",
    srcs = ["symbolic_test.cc"],
    copts = [
        "-std=c++1z",
    ],
    deps = [
        ":symbolic",
        ":symbolic_util",
        "//math/third_party:catch2",
    ],
)

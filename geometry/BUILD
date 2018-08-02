package(
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "matrix4",
    srcs = ["matrix4.cc"],
    hdrs = [
        "matrix4.h",
        "types.h",
    ],
    visibility = ["//math"],
    deps = [
        ":vector",
    ],
)

cc_library(
    name = "matrix",
    hdrs = [
        "matrix.h",
    ],
    copts = [
        "-std=c++1z",
    ],
    visibility = ["//math"],
)

cc_library(
    name = "dynamic_matrix",
    hdrs = [
        "dynamic_matrix.h",
    ],
    copts = [
        "-std=c++1z",
    ],
    visibility = ["//math"],
)

cc_binary(
    name = "matrix_test",
    srcs = ["matrix_test.cc"],
    copts = [
        "-std=c++1z",
    ],
    deps = [
        ":matrix",
        "//math/symbolic",
    ],
)

cc_binary(
    name = "dynamic_matrix_test",
    srcs = ["dynamic_matrix_test.cc"],
    copts = [
        "-std=c++1z",
    ],
    deps = [
        ":dynamic_matrix",
        "//math/symbolic",
    ],
)

cc_library(
    name = "vector",
    srcs = ["vector.cc"],
    hdrs = [
        "types.h",
        "vector.h",
    ],
    visibility = ["//math"],
)

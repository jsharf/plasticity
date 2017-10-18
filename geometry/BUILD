package(
  default_visibility = ["//visibility:public"],
)

cc_library(
  name = "matrix4",
  visibility = ["//math"],
  hdrs = [
      "matrix4.h",
      "types.h",
  ],
  srcs = ["matrix4.cc"],
  deps = [
      ":vector",
  ],
)

cc_library(
    name = "matrix",
    visibility = ["//math"],
    hdrs = [
        "matrix.h"
    ],
    copts = [
        "-std=c++1z",
    ],
)

cc_binary(
    name = "matrix_test",
    srcs = ["matrix_test.cc"],
    deps = [
        ":matrix",
        "//math/symbolic:symbolic",
    ],
    copts = [
        "-std=c++1z",
    ],
)

cc_library(
  name = "vector",
  visibility = ["//math"],
  hdrs = [
      "vector.h",
      "types.h",
  ],
  srcs = ["vector.cc"],
)

cc_library(
    name = "kalman_filter",
    visibility = ["//math"],
    hdrs = ["kalman_filter.h"],
    copts = [
        "-std=c++1z",
    ],
    deps = [
        "//math/geometry:matrix",
        "//math/symbolic:symbolic",
    ],
)

cc_binary(
  name = "kalman_test",
  srcs = ["kalman_filter_test.cc"],
  copts = [
    "-std=c++1z",
  ],
  deps = [
      ":kalman_filter",
      "//math/geometry:matrix",
      "//math/symbolic:symbolic",
  ],
)


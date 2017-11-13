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

cc_library(
    name = "ekf",
    visibility = ["//math"],
    hdrs = ["ekf.h"],
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

cc_binary(
  name = "simple_imu_test",
  srcs = ["simple_imu_test.cc"],
  copts = [
    "-std=c++1z",
    "-I/Library/Frameworks/SDL2.framework/Headers",
  ],
  deps = [
      ":kalman_filter",
      "//math/geometry:matrix",
      "//math/geometry:matrix4",
      "//math/geometry:vector",
      "//math/symbolic:symbolic",
      "//graphics:sdl_canvas",
      "//graphics:scene",
      "//graphics:perspective_camera",
      "//graphics:camera_3d",
      "//graphics:types",
      "//graphics:utah_data",
  ],
)


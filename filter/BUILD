cc_library(
    name = "kalman_filter",
    hdrs = ["kalman_filter.h"],
    copts = [
        "-std=c++1z",
    ],
    visibility = ["//plasticity"],
    deps = [
        "//plasticity/geometry:matrix",
        "//plasticity/symbolic",
    ],
)

cc_library(
    name = "ekf",
    hdrs = ["ekf.h"],
    copts = [
        "-std=c++1z",
    ],
    visibility = ["//plasticity"],
    deps = [
        "//plasticity/geometry:matrix",
        "//plasticity/symbolic",
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
        "//plasticity/geometry:matrix",
        "//plasticity/symbolic",
    ],
)

cc_binary(
    name = "simple_imu_test",
    srcs = ["simple_imu_test.cc"],
    copts = [
        "-std=c++1z",
    ],
    deps = [
        ":kalman_filter",
        "//graphics:camera_3d",
        "//graphics:perspective_camera",
        "//graphics:scene",
        "//graphics:sdl_canvas",
        "//graphics:types",
        "//graphics:utah_data",
        "//plasticity/geometry:matrix",
        "//plasticity/geometry:matrix4",
        "//plasticity/geometry:vector",
        "//plasticity/symbolic",
        "@linux_sdl//:sdl2_headers",
    ],
)

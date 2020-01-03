cc_library(
    name = "codegen",
    srcs = ["codegen.cc"],
    hdrs = ["codegen.h"],
    copts = [
        "--std=c++1z",
    ],
    visibility = ["//plasticity"],
    deps = [
    ],
)

cc_library(
    name = "codegen_util",
    hdrs = ["codegen_util.h"],
    srcs = ["codegen_util.cc"],
    copts = [
        "--std=c++1z",
    ],
    visibility = ["//plasticity"],
    deps = [
        ":codegen",
    ],
)

cc_test(
    name = "codegen_test",
    srcs = ["codegen_test.cc"],
    copts = [
        "-std=c++1z",
    ],
    deps = [
        ":codegen",
        "//plasticity/third_party:catch2",
    ],
)

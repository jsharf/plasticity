cc_library(
    name = "codegen",
    srcs = ["codegen.cc"],
    hdrs = ["codegen.h"],
    copts = [
        "--std=c++1z",
    ],
    visibility = ["//math"],
    deps = [
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
        "//math/third_party:catch2",
    ],
)

# External dependencies.
#
#new_http_archive(
#    name = "cifar-10-repo",
#    url = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz",
#    sha256 = "c4a38c50a1bc5f3a1c5537f2155ab9d68f9f25eb1ed8d9ddda3db29a59bca1dd",
#    build_file = "nnet/data/BUILD.cifar-10",
#)

filegroup(
  name = "cifar-10",
  visibility = ["//math"],
  srcs = glob(["cifar-10-batches-bin/*.bin"]),
)

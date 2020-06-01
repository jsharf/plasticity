load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

git_repository(
  name = "clutil",
  remote = "https://github.com/jsharf/clutil.git",
  commit = "5b05715412458f05fc2731dbc27621042499f434",
  shallow_since = "1578262118 -0800",
)

git_repository(
  name = "graphics",
  remote = "https://github.com/jsharf/graphics.git",
  commit = "01db82e53e437919404c2f370c7d7170fb73405a",
  shallow_since = "1590998638 -0700"
)

load("@graphics//:workspace.bzl", graphics = "repo")
graphics()

#
# SDL
#
http_archive(
    name = 'linux_sdl',
    urls = [
      'https://www.libsdl.org/release/SDL2-2.0.7.tar.gz',
    ],
    build_file = '@//:BUILD.sdl2',
    strip_prefix = 'SDL2-2.0.7',
    sha256 = "ee35c74c4313e2eda104b14b1b86f7db84a04eeab9430d56e001cea268bf4d5e",
)

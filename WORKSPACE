load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

git_repository(
  name = "clutil",
  remote = "https://github.com/jsharf/clutil.git",
  commit = "5b05715412458f05fc2731dbc27621042499f434",
  shallow_since = "1578262118 -0800",
)

git_repository(
  name = "graphics",
  remote = "https://github.com/jsharf/graphics.git",
  commit = "12221472376747a8bb8cbd31bfe1d578f777e107",
)

new_local_repository(
  name = "osx_sdl",
  path = "/Library/Frameworks/SDL2.framework/Headers",
  build_file = "BUILD.sdl",
)

new_local_repository(
  name = "linux_sdl",
  path = "/usr/include/SDL2",
  build_file = "BUILD.sdl",
)

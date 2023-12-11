project('plasticity', 'cpp', default_options: ['cpp_std=c++1z'])

root_inc = include_directories('.')

# Libraries.
symbolic_lib = library('symbolic',
    sources: [
        'symbolic/expression.cc',
        'symbolic/integer.cc',
        'symbolic/numeric_value.cc'
    ],
    include_directories: root_inc,
    install: true
)

symbolic_util_lib = library('symbolic_util',
  sources: ['symbolic/symbolic_util.cc'],
  link_with: [symbolic_lib],
  include_directories: root_inc,
  install: true
)

matrix4_lib = library('matrix4',
  sources: ['geometry/matrix4.cc'],
  include_directories: root_inc,
  install: true
)

vector_lib = library('vector',
  sources: ['geometry/vector.cc'],
  include_directories: root_inc,
  install: true
)

# Dependencies.
symbolic_dep = declare_dependency(
    link_with: symbolic_lib,
    include_directories: root_inc,
)

symbolic_util_dep = declare_dependency(
  link_with: [symbolic_util_lib],
  include_directories: root_inc
)

matrix4_dep = declare_dependency(
  link_with: [matrix4_lib],
  include_directories: root_inc
)

matrix_dep = declare_dependency(
  include_directories: root_inc
)

dynamic_matrix_dep = declare_dependency(
  include_directories: root_inc
)

vector_dep = declare_dependency(
  link_with: vector_lib,
  include_directories: root_inc
)

geometry_dep = declare_dependency(
  link_with: [matrix4_lib, vector_lib],
  include_directories: root_inc
)

# Test executables
matrix_test_exe = executable('matrix_test',
  sources: ['geometry/matrix_test.cc'],
  dependencies: [matrix_dep, symbolic_dep],
  install: false
)

dynamic_matrix_test_exe = executable('dynamic_matrix_test',
  sources: ['geometry/dynamic_matrix_test.cc'],
  dependencies: [dynamic_matrix_dep, symbolic_dep],
  install: false
)

# Define unit tests
test('matrix_test', matrix_test_exe)
test('dynamic_matrix_test', dynamic_matrix_test_exe)
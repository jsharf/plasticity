#include "nnet/symbol_generator.h"

namespace nnet {
namespace internal {

size_t Flatten2d(size_t width, size_t height, size_t row, size_t col) {
  return row * width + col;
}

size_t Unflatten2dRow(size_t width, size_t height, size_t i) {
  return i / width;
}

size_t Unflatten2dCol(size_t width, size_t height, size_t i) {
  return i % width;
}

// Assumes each filter gets serialized into row-order flattened index. Then
// filters from 0 to num_filters are appended.
// Take nnet::Dimensions instead of width, height. Handle bias inside of flatten
// functions.
size_t Flatten3d(size_t width, size_t height, size_t depth, size_t row,
                 size_t col, size_t z) {
  size_t z_plane_size = width * height;
  return z_plane_size * z + row * width + col;
}


}  // namespace internal
}  // namespace nnet

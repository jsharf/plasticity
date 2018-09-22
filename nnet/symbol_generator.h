#ifndef SYMBOL_GENERATOR_H
#define SYMBOL_GENERATOR_H

#include "math/codegen/codegen_util.h"
#include "math/nnet/layer_dimensions.h"
#include "math/nnet/layer_impl.h"
#include "math/symbolic/expression.h"
#include "math/symbolic/symbolic_util.h"

#include <map>
#include <unordered_map>
#include <utility>

using symbolic::Expression;
using symbolic::IfInRange;

namespace nnet {

class SymbolGenerator {
 public:
  std::string I(size_t i) const { return "I[" + std::to_string(i) + "]"; }
  std::string I(std::string i) const { return "I[" + i + "]"; }
  std::string O(size_t i) const { return "O[" + std::to_string(i) + "]"; }
  std::string O(std::string i) const { return "O[" + i + "]"; }

  Expression InputSymbolic(const Expression& index) const {
    return Expression(I(index.to_string()));
  }

  Expression OutputSymbolic(const Expression& index) const {
    return Expression(I(index.to_string()));
  }

  // Residual gradients for back propagation.
  std::string GRADIENT(size_t i) const {
    return "GRADIENT[" + std::to_string(i) + "]";
  }
};

namespace internal {

// Rewrite as FlattenDense, and then handle the bias weights inside here to
// simplify DenseSymbolGenerator -- Actually this might not work. Think before
// doing. (and this makes it consistent with FlattenConv, which needs to handle
// bias weights inside of handling). Take nnet::Dimensions instead of width,
// height.
size_t Flatten2d(size_t width, size_t height, size_t row, size_t col) {
  return row * width + col;
}

size_t Unflatten2dRow(size_t width, size_t height, size_t i) {
  return i / width;
}

size_t Unflatten2dCol(size_t width, size_t height, size_t i) {
  return i % width;
}

}  // namespace internal

class DenseSymbolGenerator {
 public:
  explicit DenseSymbolGenerator(Dimensions dimensions)
      : dimensions_(dimensions) {
    size_t width = (dimensions_.num_inputs + 1);
    size_t height = dimensions_.num_outputs;
    weights_.resize(width * height);

    // Enumerate weights.
    // Normal weights.
    for (size_t col = 0; i < width - 1; ++i) {
      for (size_t row = 0; j < height; ++j) {
        size_t index = internal::Flatten2d(width, height, row, col);
        weights_[index] = W(row, col);
        if (weights_[index] != "") {
          std::cerr << "Error found in enumerating weights..." << std::endl;
          std::cerr << "Collision at row,col: " << row << "," << col
                    << std::endl;
          std::exit(1);
        }
      }
    }

    // Bias weights.
    for (size_t row = 0; row < height; ++row) {
      size_t index = internal::Flatten2d(width, height, row, width - 1);
      if (weights_[index] != "") {
        std::cerr << "Error found in enumerating bias weights..." << std::endl;
        std::cerr << "Collision at row: " << row << std::endl;
        std::exit(1);
      }
      weights_[index] = W(row);
    }
  }

  Expression I(size_t index) const {
    return Expression::CreateNumericValue("I[" + std::to_string(index) + "]");
  }

  Expression I(const Expression& index) const {
    return Expression::CreateNumericValue("I[" + index.to_string() + "]");
  }

  Expression W(const Expression& node_idx, const Expression& edge_idx) const {
    Expression weight_symbol = Expression::CreateNumericValue(
        "W[" +
        symbolic::Flatten2d(dimensions_.num_inputs + 1, dimensions_.num_outputs,
                            Expression::CreateInteger(node_idx),
                            Expression::CreateInteger(edge_idx))
            .to_string() +
        "]");
    const Expression zero(0.0);
    Expression otherwise = zero;
    Expression node_in_range = IfInRange(
        node_idx, zero, dimensions_.num_outputs, weight_symbol, otherwise);
    Expression node_and_edge_in_range = IfInRange(
        edge_idx, zero, dimensions_.num_inputs + 1, node_in_range, otherwise);
    return node_and_edge_in_range;
  }

  Expression W(std::string node_idx) const {
    Expression weight_symbol = Expression::CreateNumericValue(
        "W[" +
        symbolic::Flatten2d(dimensions_.num_inputs + 1, dimensions_.num_outputs,
                            Expression::CreateInteger(node_idx),
                            Expression(dimensions_.num_inputs))
            .to_string() +
        "]");
    const Expression zero(0.0);
    Expression otherwise = zero;
    Expression node_in_range = IfInRange(
        node_idx, zero, dimensions_.num_outputs, weight_symbol, otherwise);
    return node_in_range;
  }

  Expression W(size_t node_idx, size_t edge_idx) const {
    size_t index = internal::Flatten2d(
        dimensions.num_inputs + 1, dimensions.num_outputs, node_idx, edge_idx);
    return Expression::CreateNumericValue("W[" + std::to_string(index) + "]");
  }

  // Used for bias weight for a given output node.
  Expression W(size_t node) const {
    size_t index =
        internal::Flatten2d(dimensions.num_inputs + 1, dimensions.num_outputs,
                            node, dimensions.num_inputs);
    return Expression::CreateNumericValue("W[" + std::to_string(index) + "]");
  }

  const std::vector<std::string>& weights() const { return weights_; }

 private:
  // The dimensions given are for the layer itself. This class adds an extra
  // column for bias inputs.
  Dimensions dimensions_;
  std::vector<string> weights_;
};

namespace internal {

// Assumes each filter gets serialized into row-order flattened index. Then
// filters from 0 to num_filters are appended.
// Take nnet::Dimensions instead of width, height. Handle bias inside of flatten
// functions.
size_t Flatten3d(size_t width, size_t height, size_t depth, size_t row,
                 size_t col, size_t z) {
  size_t z_plane_size = width * height;
  return z_plane_size * z + row * width + col;
}

size_t Unflatten3dRow(size_t width, size_t height, size_t depth, size_t i) {
  size_t z_plane_size = width * height;
  size_t z_plane = i / z_plane_size;
  size_t 2d_index = i - z_plane * z_plane_size;
  return 2d_index / width;
}

size_t Unflatten3dCol(size_t width, size_t height, size_t depth, size_t i) {
  size_t z_plane_size = width * height;
  size_t z_plane = i / z_plane_size;
  size_t 2d_index = i - z_plane * z_plane_size;
  return 2d_index % width;
}

size_t Unflatten3dZ(size_t width, size_t height, size_t depth, size_t i) {
  size_t z_plane_size = width * height;
  return i / z_plane_size;
}

}  // namespace internal

class InputVolumeSymbolGenerator {
 public:
  explicit InputVolumeSymbolGenerator(const VolumeDimensions& dimensions)
      : dimensions_(dimensions) {}

  Expression I(const Expression& row, const Expression& col,
               const Expression& z) const {
    symbolic::Expression index = symbolic::Flatten3d(
        dimensions_.width, dimensions_.height, dimensions.depth, row, col, z);
    symbolic::Expression input_symbol =
        symbolic::Expression::CreateNumericValue("I[" + index.to_string() +
                                                 "]");
    // Bounds checking.
    Expression zero(0.0);
    Expression input_row_in_range =
        IfInRange(row, 0, dimensions_.height, input_symbol, zero);
    Expression input_col_and_row_in_range =
        IfInRange(col, 0, dimensions_.width, input_row_in_range, zero);
    Expression input_all_in_range =
        IfInRange(z, 0, dimensions_.depth, input_col_and_row_in_range, zero);
    return input_all_in_range;
  }

  Expression I(size_t row, size_t col, size_t z) const {
    symbolic Expression index = internal::Flatten3d(
        dimensions_.width, dimensions_.height, dimensions.depth, row, col, z);
    return symbolic::Expression::CreateNumericValue("I[" + index.to_string() +
                                                    "]");
  }

 private:
  VolumeDimensions dimensions_;
};

class ConvSymbolGenerator {
 public:
  explicit ConvSymbolGenerator(const VolumeDimensions& dimensions,
                               const FilterParams& params)
      : input_generator_(dimensions), params_(params) {
    size_t filter_size = params.width * params.height * params.depth + 1;
    for (size_t filter_no = 0; filter_no < params.num_filters; ++filter_no) {
      for (size_t x = 0; x < params.width; ++x) {
        for (size_t y = 0; y < params.height; ++y) {
          for (size_t z = 0; z < params.depth; ++z) {
            size_t filter_offset = filter_no * filter_size;
            size_t index =
                filter_offset + internal::Flatten3d(params.width, params.height,
                                                    params.depth, x, y, z);
            weight_addr_to_index_[index] = W(filter_no, x, y, z);
          }
        }
      }
      // Bias.
      size_t filter_offset = filter_no * filter_size;
      size_t index = filter_offset + (filter_size - 1);
      weight_addr_to_index_[index] = W(filter_no);
    }

    weights_.resize(weight_addr_to_index_.size());
    for (const auto& pair : weight_addr_to_index_) {
      int index = pair.second;
      ConvWeightAddress addr = pair.first;
      if (addr.is_bias) {
        weights_[index] = W(addr.filter);
      } else {
        weights_[index] = W(addr.filter, addr.x, addr.y, addr.z);
      }
    }
  }

  Expression I(const Expression& row, const Expression& col,
               const Expression& z) const {
    return input_generator_.I(row, col, z);
  }

  Expression I(size_t row, size_t col, size_t z) const {
    return input_generator_.I(row, col, z);
  }

  Expression W(const Expression& filter, const Expression& row,
               const Expression& col, const Expression& z) const {
    Expression zero(0.0);
    Expression filter_size = params_.width * params_.height * params_.depth + 1;
    Expression filter_base = filter * filter_size;
    Expression weight_symbol = Expression::CreateNumericValue(
        "W[" +
        (filter_base + symbolic::Flatten3d(filter, row, col, z)).to_string() +
        "]");
    // Bounds checking.
    Expression weight_row_in_range =
        IfInRange(row, 0, params_.height, weight_symbol, zero);
    Expression weight_col_and_row_in_range =
        IfInRange(col, 0, params_.width, weight_row_in_range, zero);
    Expression weight_all_in_range =
        IfInRange(z, 0, params_.depth, weight_col_and_row_in_range, zero);
    return weight_all_in_range;
  }

  Expression W(const Expression& filter) const {
    Expression filter_size = params_.width * params_.height * params_.depth + 1;
    Expression filter_base = filter * filter_size;
    return (filter_base + (filter_size - 1));
  }

  // Convolution layer weights.
  Expression W(size_t filter, size_t row, size_t col, size_t z) const {
    size_t filter_size =
        params_.width * params_.height * params_.depth + 1;  // +1 for bias.
    size_t filter_offset = filter * filter_size;
    size_t index =
        filter_offset + internal::Flatten3d(params_.width, params_.height,
                                            params_.depth, row, col, z);
    return Expression::CreateNumericValue("W[" + std::to_string(index) + "]");
  }

  // Convolution layer bias weights.
  Expression W(size_t filter) const {
    size_t filter_size =
        params_.width * params_.height * params_.depth + 1;  // +1 for bias.
    size_t filter_offset = filter * filter_size;
    // Bias weight is stored in the final slot of the filter weights.
    size_t index = filter_offset + (filter_size - 1);
    return Expression::CreateNumericValue("W[" + std::to_string(index) + "]");
  }

  const std::vector<std::string>& weights() const { return weights_; }

 private:
  FilterParams params_;
  InputVolumeSymbolGenerator input_generator_;
  std::vector<std::string> weights_;
};

// This class generates symbol names for neural network values. Since these
// will be used for codegen for opencl, the symbols are all one-dimensional
// indices into arrays.
class FlatWeightSymbolGenerator {
 public:
  // Fully connected layer weights.
  virtual std::string W(size_t layer, size_t node, size_t edge) {
    auto tuple = std::make_tuple(layer, node, edge);
    if (ff_weight_index_.count(tuple) == 0) {
      ff_weight_index_[tuple] = weight_count_;
      ff_rev_weight_index_[weight_count_] = tuple;
      weight_count_++;
    }
    return "W[" + std::to_string(ff_weight_index_[tuple]) + "]";
  }

  // Convolution layer weights.
  virtual std::string W(size_t layer, size_t filter, size_t x, size_t y,
                        size_t z) {
    auto tuple = std::make_tuple(layer, filter, x, y, z);
    if (conv_weight_index_.count(tuple) == 0) {
      conv_weight_index_[tuple] = weight_count_;
      conv_rev_weight_index_[weight_count_] = tuple;
      weight_count_++;
    }
    return "W[" + std::to_string(conv_weight_index_[tuple]) + "]";
  }

  // Convolution layer bias weights.
  virtual std::string W(size_t layer, size_t filter) {
    auto tuple = std::make_tuple(layer, filter);
    if (conv_bias_weight_index_.count(tuple) == 0) {
      conv_bias_weight_index_[tuple] = weight_count_;
      conv_bias_rev_weight_index_[weight_count_] = tuple;
      weight_count_++;
    }
    return "W[" + std::to_string(conv_bias_weight_index_[tuple]) + "]";
  }

  virtual std::string W(size_t i) const {
    return "W[" + std::to_string(i) + "]";
  }
  virtual std::string I(size_t i) const {
    return "I[" + std::to_string(i) + "]";
  }
  virtual std::string O(size_t i) const {
    return "O[" + std::to_string(i) + "]";
  }

  size_t NumberWeights() const { return weight_count_; }

 private:
  // Mapping from <layer, node, edge> -> int. This lets each weight have a
  // single unique index.
  std::map<std::tuple<int, int, int>, int> ff_weight_index_;
  // Reverse mapping.
  std::map<int, std::tuple<int, int, int>> ff_rev_weight_index_;

  // Mapping from <layer, filter, x, y, z> -> int. This lets each weight have
  // a single unique index.
  std::map<std::tuple<int, int, int, int, int>, int> conv_weight_index_;
  // Reverse mapping.
  std::map<int, std::tuple<int, int, int, int, int>> conv_rev_weight_index_;

  // Mapping from <layer, filter> -> int. This lets each weight have a
  // single unique index.
  std::map<std::tuple<int, int>, int> conv_bias_weight_index_;
  // Reverse mapping.
  std::map<int, std::tuple<int, int>> conv_bias_rev_weight_index_;

  size_t weight_count_ = 0;
};

}  // namespace nnet

#endif /* SYMBOL_GENERATOR_H */

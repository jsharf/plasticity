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

  Expression I(const Expression &index) const {
    return Expression::CreateNumericValue(I(index.to_string()));
  }

  Expression O(const Expression &index) const {
    return Expression::CreateNumericValue(I(index.to_string()));
  }

  // Residual gradients for back propagation.
  std::string GRADIENT(size_t i) const {
    return "GRADIENT[" + std::to_string(i) + "]";
  }

  Expression GRADIENT(const symbolic::Expression &i) const {
    return Expression::CreateNumericValue("GRADIENT[" + i.to_string() + "]");
  }
};

namespace internal {

size_t Flatten2d(size_t width, size_t height, size_t row, size_t col);
size_t Unflatten2dRow(size_t width, size_t height, size_t i);
size_t Unflatten2dCol(size_t width, size_t height, size_t i);
size_t Flatten3d(size_t width, size_t height, size_t depth, size_t row,
                 size_t col, size_t z);

} // namespace internal

class DenseSymbolGenerator {
public:
  explicit DenseSymbolGenerator(Dimensions dimensions)
      : dimensions_(dimensions) {
    size_t width = (dimensions_.num_inputs + 1);
    size_t height = dimensions_.num_outputs;
    weights_.resize(width * height);

    // Enumerate weights.
    // Normal weights.
    for (size_t col = 0; col < width - 1; ++col) {
      for (size_t row = 0; row < height; ++row) {
        size_t index = internal::Flatten2d(width, height, row, col);
        if (weights_[index] != "") {
          std::cerr << "Error found in enumerating weights..." << std::endl;
          std::cerr << "Collision at row,col: " << row << "," << col
                    << std::endl;
          std::exit(1);
        }
        weights_[index] = W(row, col).to_string();
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
      weights_[index] = W(row).to_string();
    }
  }

  Expression I(size_t index) const {
    return Expression::CreateNumericValue("I[" + std::to_string(index) + "]");
  }

  Expression I(const Expression &index) const {
    return Expression::CreateNumericValue("I[" + index.to_string() + "]");
  }

  Expression W(const Expression &node_idx, const Expression &edge_idx) const {
    return Expression::CreateNumericValue(
        "W[" +
        symbolic::Flatten2d(dimensions_.num_inputs + 1, dimensions_.num_outputs,
                            node_idx, edge_idx)
            .to_string() +
        "]");
  }

  Expression BoundsCheckedW(const Expression &node_idx,
                            const Expression &edge_idx) const {
    Expression weight_symbol = W(node_idx, edge_idx);
    const Expression zero(0.0);
    Expression otherwise = zero;
    Expression node_in_range = IfInRange(
        node_idx, zero, dimensions_.num_outputs, weight_symbol, otherwise);
    Expression node_and_edge_in_range = IfInRange(
        edge_idx, zero, dimensions_.num_inputs + 1, node_in_range, otherwise);
    return node_and_edge_in_range;
  }

  Expression W(const Expression &node_idx) const {
    return Expression::CreateNumericValue(
        "W[" +
        symbolic::Flatten2d(dimensions_.num_inputs + 1, dimensions_.num_outputs,
                            node_idx, Expression(dimensions_.num_inputs))
            .to_string() +
        "]");
  }

  Expression BoundsCheckedW(const Expression &node_idx) const {
    Expression weight_symbol = W(node_idx);
    const Expression zero(0.0);
    Expression otherwise = zero;
    Expression node_in_range = IfInRange(
        node_idx, zero, dimensions_.num_outputs, weight_symbol, otherwise);
    return node_in_range;
  }

  Expression W(size_t node_idx, size_t edge_idx) const {
    return Expression::CreateNumericValue(
        "W[" + std::to_string(WeightNumber(node_idx, edge_idx)) + "]");
  }

  size_t WeightNumber(size_t node_idx, size_t edge_idx) const {
    return internal::Flatten2d(dimensions_.num_inputs + 1,
                               dimensions_.num_outputs, node_idx, edge_idx);
  }

  // Used for bias weight for a given output node.
  Expression W(size_t node) const {
    return Expression::CreateNumericValue(
        "W[" + std::to_string(WeightNumber(node)) + "]");
  }

  size_t WeightNumber(size_t node) const {
    return internal::Flatten2d(dimensions_.num_inputs + 1,
                               dimensions_.num_outputs, node,
                               dimensions_.num_inputs);
  }

  // Residual gradients for back propagation.
  Expression GRADIENT(size_t i) const {
    return Expression::CreateNumericValue("GRADIENT[" + std::to_string(i) +
                                          "]");
  }

  Expression GRADIENT(const symbolic::Expression &i) const {
    return Expression::CreateNumericValue("GRADIENT[" + i.to_string() + "]");
  }

  const std::vector<std::string> &weights() const { return weights_; }

private:
  // The dimensions given are for the layer itself. This class adds an extra
  // column for bias inputs.
  Dimensions dimensions_;
  std::vector<string> weights_;
};

class InputVolumeSymbolGenerator {
public:
  explicit InputVolumeSymbolGenerator(const VolumeDimensions &dimensions)
      : dimensions_(dimensions) {}

  Expression I(const Expression &row, const Expression &col,
               const Expression &z) const {
    symbolic::Expression index = symbolic::Flatten3d(
        dimensions_.width, dimensions_.height, dimensions_.depth, row, col, z);
    return symbolic::Expression::CreateNumericValue("I[" + index.to_string() +
                                                    "]");
  }

  Expression GRADIENT(size_t i) const {
    return Expression::CreateNumericValue("GRADIENT[" + std::to_string(i) +
                                          "]");
  }

  Expression GRADIENT(const symbolic::Expression &i) const {
    return Expression::CreateNumericValue("GRADIENT[" + i.to_string() + "]");
  }

  Expression BoundsCheckedI(const Expression &row, const Expression &col,
                            const Expression &z) const {
    Expression input_symbol = I(row, col, z);
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
    symbolic::Expression index = internal::Flatten3d(
        dimensions_.width, dimensions_.height, dimensions_.depth, row, col, z);
    return symbolic::Expression::CreateNumericValue("I[" + index.to_string() +
                                                    "]");
  }

private:
  VolumeDimensions dimensions_;
};

class ConvSymbolGenerator {
public:
  explicit ConvSymbolGenerator(const VolumeDimensions &dimensions,
                               const FilterParams &params)
      : input_generator_(dimensions), params_(params) {
    size_t filter_size = params.width * params.height * params.depth + 1;
    // +1 for bias.
    weights_.resize(params.num_filters * (params.width * params.height * params.depth + 1));
    for (size_t filter_no = 0; filter_no < params.num_filters; ++filter_no) {
      for (size_t x = 0; x < params.width; ++x) {
        for (size_t y = 0; y < params.height; ++y) {
          for (size_t z = 0; z < params.depth; ++z) {
            size_t filter_offset = filter_no * filter_size;
            size_t index =
                filter_offset + internal::Flatten3d(params.width, params.height,
                                                    params.depth, x, y, z);
            weights_[index] = W(filter_no, x, y, z).to_string();
          }
        }
      }
      // Bias.
      size_t filter_offset = filter_no * filter_size;
      size_t index = filter_offset + (filter_size - 1);
      weights_[index] = W(filter_no).to_string();
    }

  }

  Expression GRADIENT(size_t i) const {
    return Expression::CreateNumericValue("GRADIENT[" + std::to_string(i) +
                                          "]");
  }

  Expression GRADIENT(const symbolic::Expression &i) const {
    return Expression::CreateNumericValue("GRADIENT[" + i.to_string() + "]");
  }

  Expression I(const Expression &row, const Expression &col,
               const Expression &z) const {
    return input_generator_.I(row, col, z);
  }

  Expression BoundsCheckedI(const Expression &row, const Expression &col,
                            const Expression &z) const {
    return input_generator_.BoundsCheckedI(row, col, z);
  }

  Expression W(const Expression &filter, const Expression &row,
               const Expression &col, const Expression &z) const {
    Expression zero(0.0);
    // +1 for bias value.
    Expression filter_size = params_.width * params_.height * params_.depth + 1;
    Expression filter_base = filter * filter_size;
    return Expression::CreateNumericValue(
        "W[" +
        (filter_base + symbolic::Flatten3d(params_.width, params_.height,
                                           params_.depth, row, col, z))
            .to_string() +
        "]");
  }

  Expression GetWeightFilter(const Expression& flat_index) const {
    Expression filter_size = params_.width * params_.height * params_.depth + 1;
    return flat_index / filter_size;
  }

  Expression GetWeightX(const Expression& flat_index) const {
    Expression filter_size = params_.width * params_.height * params_.depth + 1;
    Expression weight_offset = flat_index % filter_size;
    return symbolic::Unflatten3dCol(params_.width, params_.height,
                                    params_.depth, weight_offset);
  }

  Expression GetWeightY(const Expression& flat_index) const {
    Expression filter_size = params_.width * params_.height * params_.depth + 1;
    Expression weight_offset = flat_index % filter_size;
    return symbolic::Unflatten3dRow(params_.width, params_.height,
                                    params_.depth, weight_offset);
  }

  Expression GetWeightZ(const Expression& flat_index) const {
    Expression filter_size = params_.width * params_.height * params_.depth + 1;
    Expression weight_offset = flat_index % filter_size;
    return symbolic::Unflatten3dPlane(params_.width, params_.height,
                                      params_.depth, weight_offset);
  }

  Expression BoundsCheckedW(const Expression &filter, const Expression &row,
                            const Expression &col, const Expression &z) const {
    Expression weight_symbol = W(filter, row, col, z);
    // Bounds checking.
    Expression weight_filter_in_range =
        IfInRange(filter, 0, params_.num_filters, weight_symbol, 0);
    Expression weight_row_in_range =
        IfInRange(row, 0, params_.height, weight_filter_in_range, 0);
    Expression weight_col_and_row_in_range =
        IfInRange(col, 0, params_.width, weight_row_in_range, 0);
    Expression weight_all_in_range =
        IfInRange(z, 0, params_.depth, weight_col_and_row_in_range, 0);
    return weight_all_in_range;
  }

  Expression W(const Expression &filter) const {
    Expression filter_size = params_.width * params_.height * params_.depth + 1;
    Expression filter_base = filter * filter_size;
    return Expression::CreateNumericValue(
        "W[" + (filter_base + (filter_size - 1)).to_string() + "]");
  }

  // Convolution layer weights.
  Expression W(size_t filter, size_t row, size_t col, size_t z) const {
    return Expression::CreateNumericValue(
        "W[" + std::to_string(WeightNumber(filter, row, col, z)) + "]");
  }

  size_t WeightNumber(size_t filter, size_t row, size_t col, size_t z) const {
    size_t filter_size =
        params_.width * params_.height * params_.depth + 1; // +1 for bias.
    size_t filter_offset = filter * filter_size;
    return filter_offset + internal::Flatten3d(params_.width, params_.height,
                                               params_.depth, row, col, z);
  }

  // Bias weight.
  size_t WeightNumber(size_t filter) const {
    size_t filter_size =
        params_.width * params_.height * params_.depth + 1; // +1 for bias.
    size_t filter_offset = filter * filter_size;
    // Bias weight is stored in the final slot of the filter weights.
    return filter_offset + (filter_size - 1);
  }

  // Convolution layer bias weights.
  Expression W(size_t filter) const {
    return Expression::CreateNumericValue(
        "W[" + std::to_string(WeightNumber(filter)) + "]");
  }

  const std::vector<std::string> &weights() const { return weights_; }

private:
  InputVolumeSymbolGenerator input_generator_;
  FilterParams params_;
  std::vector<std::string> weights_;
};

} // namespace nnet

#endif /* SYMBOL_GENERATOR_H */

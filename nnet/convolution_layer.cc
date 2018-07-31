#include "math/nnet/convolution_layer.h"

#include <cassert>

namespace nnet {

ConvolutionLayer::ConvolutionLayer(const VolumeDimensions& dimensions,
                                   const FilterParams& filters,
                                   IndexMap input_map, IndexMap output_map,
                                   size_t layer_index)
    : Super(GenLinearDimensions(dimensions, filters), layer_index),
      generator_(filters),
      filters_(filters),
      imdim_(dimensions),
      input_map_(input_map),
      output_map_(output_map) {
  if (filters_.depth != imdim_.depth) {
    std::cerr << "Convolution layer input depth != filter depth. Error!"
              << std::endl;
    std::exit(1);
  }
}

ConvolutionLayer::ConvolutionLayer(const VolumeDimensions& dimensions,
                                   const FilterParams& filters, size_t layer_index)
    : Super(GenLinearDimensions(dimensions, filters), layer_index),
      generator_(filters),
      filters_(filters),
      imdim_(dimensions) {
  if (filters_.depth != imdim_.depth) {
    std::cerr << "Convolution layer input depth != filter depth. Error!"
              << std::endl;
    std::exit(1);
  }
  // If you don't specify the input and output map, you get these silly
  // defaults. Your fault for not caring. I should probably delete this
  // constructor to make people care. Oh well... TODO(sharf): document this
  // better.
  //
  // The defaults line up with the format of data in the CIFAR database,
  // described here:
  // https://www.cs.toronto.edu/~kriz/cifar.html
  //
  input_map_ = [this](size_t x, size_t y, size_t z) -> size_t {
    size_t row_index = y * (imdim_.width);
    size_t col_index = x;
    size_t depth_index = z * (imdim_.width * imdim_.height);
    return row_index + col_index + depth_index;
  };

  std::tuple<size_t, size_t, size_t> output_dims =
      GetOutputDimensions(imdim_, filters_);
  size_t output_width = std::get<0>(output_dims);
  size_t output_height = std::get<1>(output_dims);
  size_t output_depth = std::get<2>(output_dims);

  output_map_ = [output_width, output_height, output_depth](
                    size_t x, size_t y, size_t z) -> size_t {
    size_t row_index = y * (output_width);
    size_t col_index = x;
    size_t depth_index = z * (output_width * output_height);
    return row_index + col_index + depth_index;
  };
}

LinearDimensions ConvolutionLayer::GenLinearDimensions(
    const VolumeDimensions& dim, const FilterParams& filters) {
  std::tuple<size_t, size_t, size_t> output_dims =
      GetOutputDimensions(dim, filters);
  size_t output_width = std::get<0>(output_dims);
  size_t output_height = std::get<1>(output_dims);
  size_t output_depth = std::get<2>(output_dims);
  return LinearDimensions{
      // Num Inputs.
      dim.width * dim.height * dim.depth,
      // Num Outputs.
      output_width * output_height * output_depth,
  };
}

std::tuple<size_t, size_t, size_t> ConvolutionLayer::GetOutputDimensions(
    const VolumeDimensions& dim, const FilterParams& filters) {
  size_t output_width =
      (dim.width - filters.width + filters.padding * 2) / filters.stride + 1;
  size_t output_height =
      (dim.height - filters.height + filters.padding * 2) / filters.stride + 1;
  size_t output_depth = filters.num_filters;
  return std::make_tuple(output_width, output_height, output_depth);
}

const std::vector<std::string>& ConvolutionLayer::weights() const {
  return generator_.weights();
}

Matrix<symbolic::Expression> ConvolutionLayer::GenerateExpression(
    const Matrix<symbolic::Expression>& input) {
  auto dim = input.size();
  size_t rows = std::get<0>(dim);
  size_t cols = std::get<1>(dim);

  // Validate input dimensions.
  if ((rows != dimensions_.num_inputs) || (cols != 1)) {
    std::cerr << "Error: ConvolutionLayer::GenerateExpression called on input "
                 "of incorrect size: "
              << "(" << rows << ", " << cols << ")" << std::endl;
    std::exit(1);
  }

  // Get 3D output dimensions. (output will be a 1D serialized form of this,
  // using mapping output_map_).
  std::tuple<size_t, size_t, size_t> output_dims =
      GetOutputDimensions(imdim_, filters_);
  size_t output_width = std::get<0>(output_dims);
  size_t output_height = std::get<1>(output_dims);
  size_t output_depth = std::get<2>(output_dims);

  Matrix<symbolic::Expression> output(
      output_width * output_height * output_depth, 1);

  for (size_t filter_no = 0; filter_no < filters_.num_filters; ++filter_no) {
    int start_x = -filters_.padding;
    int start_y = -filters_.padding;
    for (size_t out_x = 0; out_x < output_width; ++out_x) {
      for (size_t out_y = 0; out_y < output_height; ++out_y) {
        symbolic::Expression convolution =
            symbolic::CreateExpression(generator_.W(filter_no));  // bias term.
        for (size_t f_x = 0; f_x < filters_.width; ++f_x) {
          for (size_t f_y = 0; f_y < filters_.height; ++f_y) {
            for (size_t f_z = 0; f_z < filters_.depth; ++f_z) {
              int input_x = start_x + (out_x * filters_.stride) + f_x;
              int input_y = start_y + (out_y * filters_.stride) + f_y;
              int input_z = f_z;
              // Make sure that we aren't indexing out-of-bounds for
              // zero-padding case.
              if ((input_x >= 0) &&
                  (input_x < static_cast<int>(imdim_.width)) &&
                  (input_y >= 0) &&
                  (input_y < static_cast<int>(imdim_.height)) &&
                  (input_z >= 0) &&
                  (input_z < static_cast<int>(imdim_.depth))) {
                convolution +=
                    input.at(input_map_(input_x, input_y, input_z), 0) *
                    symbolic::CreateExpression(
                        generator_.W(filter_no, f_x, f_y, f_z));
              }
            }
          }
        }
        output.at(output_map_(out_x, out_y, filter_no), 0) = convolution;
      }
    }
  }

  return output;
}

std::unique_ptr<LayerImpl> ConvolutionLayer::Clone() const {
  return std::make_unique<ConvolutionLayer>(imdim_, filters_, Super::layer_index_);
}

}  // namespace nnet

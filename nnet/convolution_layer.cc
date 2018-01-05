#include "math/nnet/convolution_layer.h"

#include <cassert>

namespace nnet {

ConvolutionLayer::ConvolutionLayer(const VolumeDimensions& dimensions,
                                   const FilterParams& filters,
                                   SymbolGenerator* generator,
                                   size_t layer_index)
    : Super(GenLinearDimensions(dimensions, filters), generator, layer_index),
      filters_(filters),
      imdim_(dimensions) {
  if (filters_.depth != imdim_.depth) {
    std::cerr << "Convolution layer input depth != filter depth. Error!"
              << std::endl;
    std::exit(1);
  }
}

ConvolutionLayer::LinearDimensions ConvolutionLayer::GenLinearDimensions(
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

LayerImpl::WeightArray ConvolutionLayer::weights() {
  WeightArray weights(filters_.num_filters *
                      (filters_.width * filters_.height * filters_.depth + 1));
  size_t back_index = 0;
  for (size_t filter_no = 0; filter_no < filters_.num_filters; ++filter_no) {
    for (size_t x = 0; x < filters_.width; ++x) {
      for (size_t y = 0; y < filters_.height; ++y) {
        for (size_t z = 0; z < filters_.depth; ++z) {
          assert(back_index < weights.size());
          weights[back_index++] =
              Super::generator_->W(Super::layer_index_, filter_no, x, y, z);
        }
      }
    }
    // Bias.
    weights[back_index++] =
        Super::generator_->W(Super::layer_index_, filter_no);
  }
  return weights;
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
  // using mapping output_flat_index).
  std::tuple<size_t, size_t, size_t> output_dims =
      GetOutputDimensions(imdim_, filters_);
  size_t output_width = std::get<0>(output_dims);
  size_t output_height = std::get<1>(output_dims);
  size_t output_depth = std::get<2>(output_dims);

  // Converts 3D index (x, y, z) into 1D index into input.
  // This is something I'm not proud of. The conversion between the 1D neural
  // network layers in a feed forward net and the 3D volumes in a convolution
  // layer should be better documented, but I'm settling on this for now.
  //
  // As an example of how this works, imagine an RGB image with a width W and
  // height H. One row would be serialized as W r,g,b values, as such:
  // row = [r1, g1, b1, r2, g2, b2, r3, g3, b3 .... rW, gW, bW]
  //
  // And the entire image would just be a concatenated list of H serialized
  // rows
  std::function<size_t(size_t, size_t, size_t)> input_flat_index =
      [this](size_t x, size_t y, size_t z) -> size_t {
    size_t row_index = y * (imdim_.width * imdim_.depth);
    size_t col_index = x * imdim_.depth;
    size_t depth_index = z;
    return row_index + col_index + depth_index;
  };

  //
  Matrix<symbolic::Expression> output(
      output_width * output_height * output_depth, 1);

  std::function<size_t(size_t, size_t, size_t)> output_flat_index =
      [output_width, output_height, output_depth](size_t x, size_t y,
                                                  size_t z) -> size_t {
    size_t row_index = y * (output_width * output_depth);
    size_t col_index = x * output_depth;
    size_t depth_index = z;
    return row_index + col_index + depth_index;
  };

  for (size_t filter_no = 0; filter_no < filters_.num_filters; ++filter_no) {
    int start_x = -filters_.padding;
    int start_y = -filters_.padding;
    for (size_t out_x = 0; out_x < output_width; ++out_x) {
      for (size_t out_y = 0; out_y < output_height; ++out_y) {
        symbolic::Expression convolution = symbolic::CreateExpression("0");
        for (size_t f_x = 0; f_x < filters_.width; ++f_x) {
          for (size_t f_y = 0; f_y < filters_.height; ++f_y) {
            for (size_t f_z = 0; f_z < filters_.depth; ++f_z) {
              int input_x = start_x + (out_x * filters_.stride) + f_x;
              int input_y = start_y + (out_y * filters_.stride) + f_y;
              int input_z = f_z;
              // Make sure that we aren't indexing out-of-bounds for
              // zero-padding case.
              if ((input_x >= 0) && (input_x < imdim_.width) &&
                  (input_y >= 0) && (input_y < imdim_.height) &&
                  (input_z >= 0) && (input_z < imdim_.depth)) {
                convolution +=
                    input.at(input_flat_index(input_x, input_y, input_z), 0) *
                    symbolic::CreateExpression(Super::generator_->W(
                        Super::layer_index_, filter_no, f_x, f_y, f_z));
              }
            }
          }
        }
        output.at(output_flat_index(out_x, out_y, filter_no), 0) = convolution;
      }
    }
  }

  return output;
}

stats::Normal ConvolutionLayer::XavierInitializer() const {
  // + filters_.num_filters since each filter has an implicit bias input.
  return stats::Normal(0,
                       1.0 / (dimensions_.num_inputs + filters_.num_filters));
}

}  // namespace nnet

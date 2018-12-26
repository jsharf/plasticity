#include "math/nnet/convolution_layer.h"

#include <cassert>

namespace nnet {

ConvolutionLayer::ConvolutionLayer(const VolumeDimensions& dimensions,
                                   const FilterParams& filters,
                                   size_t layer_index)
    : Super(GenLinearDimensions(dimensions, filters), layer_index),
      generator_(dimensions, filters),
      filters_(filters),
      imdim_(dimensions) {
  if (filters_.depth != imdim_.depth) {
    std::cerr << "Convolution layer input depth != filter depth. Error!"
              << std::endl;
    std::exit(1);
  }
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

symbolic::Expression ConvolutionLayer::GenerateOutputCode(
    const symbolic::Expression& index) const {
  std::tuple<size_t, size_t, size_t> output_dims =
      GetOutputDimensions(imdim_, filters_);
  size_t output_width = std::get<0>(output_dims);
  size_t output_height = std::get<1>(output_dims);
  size_t output_depth = std::get<2>(output_dims);

  symbolic::Expression output_row = symbolic::Unflatten3dRow(
      output_width, output_height, output_depth, index);

  symbolic::Expression output_col = symbolic::Unflatten3dCol(
      output_width, output_height, output_depth, index);

  symbolic::Expression output_filter = symbolic::Unflatten3dPlane(
      output_width, output_height, output_depth, index);

  // Add bias to the output.
  symbolic::Expression output = generator_.W(output_filter);

  symbolic::Expression conv_start_row =
      (output_row * filters_.stride) - filters_.padding + filters_.width / 2;
  symbolic::Expression conv_start_col =
      (output_col * filters_.stride) - filters_.padding + filters_.height / 2;

  // Sum up the convolution, adding it to the output.
  for (size_t f_x = 0; f_x < filters_.width; ++f_x) {
    for (size_t f_y = 0; f_y < filters_.height; ++f_y) {
      for (size_t f_z = 0; f_z < filters_.depth; ++f_z) {
        symbolic::Expression input_x = conv_start_row + f_x;
        symbolic::Expression input_y = conv_start_col + f_y;
        symbolic::Expression input_z = f_z;
        output += generator_.BoundsCheckedW(output_filter, f_x, f_y, f_z) *
                  generator_.BoundsCheckedI(input_x, input_y, input_z);
      }
    }
  }

  return output;
}

Matrix<std::vector<symbolic::Expression>>
ConvolutionLayer::InputGradientsForOutput(
    const symbolic::Expression& index) const {
  symbolic::Expression output = GenerateOutputCode(index);
  Matrix<std::vector<symbolic::Expression>> gradients(filters_.width,
                                                      filters_.height);

  std::tuple<size_t, size_t, size_t> output_dims =
      GetOutputDimensions(imdim_, filters_);
  size_t output_width = std::get<0>(output_dims);
  size_t output_height = std::get<1>(output_dims);
  size_t output_depth = std::get<2>(output_dims);

  symbolic::Expression output_row = symbolic::Unflatten3dRow(
      output_width, output_height, output_depth, index);

  symbolic::Expression output_col = symbolic::Unflatten3dCol(
      output_width, output_height, output_depth, index);

  symbolic::Expression output_filter = symbolic::Unflatten3dPlane(
      output_width, output_height, output_depth, index);

  symbolic::Expression conv_start_row =
      (output_row * filters_.stride) - filters_.padding + filters_.width / 2;
  symbolic::Expression conv_start_col =
      (output_col * filters_.stride) - filters_.padding + filters_.height / 2;

  for (size_t f_x = 0; f_x < filters_.width; ++f_x) {
    for (size_t f_y = 0; f_y < filters_.height; ++f_y) {
      gradients.at(f_x, f_y).resize(filters_.depth);
      for (size_t f_z = 0; f_z < filters_.depth; ++f_z) {
        symbolic::Expression input_x = conv_start_row + f_x;
        symbolic::Expression input_y = conv_start_col + f_y;
        symbolic::Expression input_z = f_z;
        gradients.at(f_x, f_y)[f_z] =
            output.Derive(generator_.I(input_x, input_y, input_z).to_string());
      }
    }
  }
  return gradients;
}

symbolic::Expression ConvolutionLayer::InputGradientCode(const symbolic::Expression& index) const {
  return symbolic::Expression(0);
}

// symbolic::Expression ConvolutionLayer::InputGradientCode(
//     const symbolic::Expression& index) const {
//   size_t input_width = imdim_.width;
//   size_t input_height = imdim_.height;
//   size_t input_depth = imdim_.depth;
// 
//   symbolic::Expression input_row =
//       symbolic::Unflatten3dRow(input_width, input_height, input_depth, index);
// 
//   symbolic::Expression input_col =
//       symbolic::Unflatten3dCol(input_width, input_height, input_depth, index);
// 
//   symbolic::Expression input_plane =
//       symbolic::Unflatten3dPlane(input_width, input_height, input_depth, index);
// 
//   symbolic::Expression output_row =
//       (input_row + filters_.padding - filters_.width / 2) / filters_.stride;
//   symbolic::Expression output_col =
//       (input_col + filters_.padding - filters_.height / 2) / filters_.stride;
// 
//   size_t output_net_width = filters_.width/(filters_.stride);
//   size_t output_net_height = filters_.height/(filters_.stride);
//   for (int d = -output_net_width/2; d <= output_net_width/2; d++) {
//     for (size_t k = -output_net_height/2; k <= output_net_height/2; k++) {
//       symbolic::Expression neighbor_output_flat_index = symbolic::Flatten3d(output_row , output_col, 
//           // Need to think about this one. I think I need to iterate over
//           // filter.num_filters since each is a diff output and then accumulate
//           // all these gradients into an expression and then return it.
//     }
//   }
// }

Matrix<symbolic::Expression> ConvolutionLayer::WeightGradientsForOutput(
    const symbolic::Expression& index) const {
      return Matrix<symbolic::Expression>();
}

std::unique_ptr<LayerImpl> ConvolutionLayer::Clone() const {
  return std::make_unique<ConvolutionLayer>(imdim_, filters_,
                                            Super::layer_index_);
}

}  // namespace nnet

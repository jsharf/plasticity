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
      (output_row * filters_.stride) - filters_.padding;
  symbolic::Expression conv_start_col =
      (output_col * filters_.stride) - filters_.padding;

  // Sum up the convolution, adding it to the output.
  for (size_t f_x = 0; f_x < filters_.width; ++f_x) {
    for (size_t f_y = 0; f_y < filters_.height; ++f_y) {
      for (size_t f_z = 0; f_z < filters_.depth; ++f_z) {
        symbolic::Expression input_x = conv_start_row + f_x;
        symbolic::Expression input_y = conv_start_col + f_y;
        symbolic::Expression input_z = f_z;
        output += generator_.W(output_filter, input_x, input_y, input_z) *
                  generator_.I(input_x, input_y, input_z);
      }
    }
  }

  return output;
}

std::unique_ptr<LayerImpl> ConvolutionLayer::Clone() const {
  return std::make_unique<ConvolutionLayer>(imdim_, filters_,
                                            Super::layer_index_);
}

}  // namespace nnet

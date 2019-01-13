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

void ConvolutionLayer::GenerateOutputCode(
    const symbolic::Expression& index, codegen::Generator *cg) const {
  symbolic::Expression retval = GenerateOutputSymbol(index);
  cg->AppendLineOfCode("return " + retval.to_string() + cg->linesep());
}

symbolic::Expression ConvolutionLayer::GenerateOutputSymbol(
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
  symbolic::Expression output = GenerateOutputSymbol(index);
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

void ConvolutionLayer::InputGradientCode(const symbolic::Expression &index,
                                         codegen::Generator *cg) const {
  std::tuple<size_t, size_t, size_t> output_dims =
      GetOutputDimensions(imdim_, filters_);
  size_t output_width = std::get<0>(output_dims);
  size_t output_height = std::get<1>(output_dims);
  size_t output_depth = std::get<2>(output_dims);
  size_t input_width = imdim_.width;
  size_t input_height = imdim_.height;
  size_t input_depth = imdim_.depth;

  symbolic::Expression input_row =
      symbolic::Unflatten3dRow(input_width, input_height, input_depth, index);

  symbolic::Expression input_col =
      symbolic::Unflatten3dCol(input_width, input_height, input_depth, index);

  symbolic::Expression input_plane =
      symbolic::Unflatten3dPlane(input_width, input_height, input_depth, index);

  symbolic::Expression output_row =
      (input_row + filters_.padding - filters_.width / 2) / filters_.stride;
  symbolic::Expression output_col =
      (input_col + filters_.padding - filters_.height / 2) / filters_.stride;

  // The "net" includes all outputs which depend on this input. This is
  // determined by the filter size and stride length. If the stride is 2, for
  // instance, then inputs are skipped during the convolution, decreasing the
  // number of outputs reliant on this input. The best way to illustrate this
  // is by visualizing the convolution.
  int output_net_width = filters_.width / (filters_.stride);
  int output_net_height = filters_.height / (filters_.stride);
  symbolic::Expression gradient_code(0.0);
  for (int d = -output_net_width / 2; d <= output_net_width / 2; d++) {
    for (int k = -output_net_height / 2; k <= output_net_height / 2; k++) {
      for (size_t filter = 0; filter < filters_.num_filters; ++filter) {
        symbolic::Expression neighbor_output_flat_index = symbolic::Flatten3d(
            output_width, output_height, output_depth, output_row + d,
            output_col + k, symbolic::Expression(filter));
        Matrix<std::vector<symbolic::Expression>> neighbor_gradients =
            InputGradientsForOutput(neighbor_output_flat_index);
        const std::vector<symbolic::Expression> &self_gradients_of_neighbor =
            neighbor_gradients.at(output_net_width / 2 - d,
                                  output_net_height / 2 - k);
        for (const auto &gradient_component : self_gradients_of_neighbor) {
          gradient_code += gradient_component *
                           generator_.GRADIENT(neighbor_output_flat_index);
        }
      }
    }
  }
  cg->AppendLineOfCode("return " + gradient_code.to_string() + cg->linesep());
}

void ConvolutionLayer::WeightGradientCode(
    const symbolic::Expression &index, codegen::Generator *cg) const {
  std::tuple<size_t, size_t, size_t> output_dims =
      GetOutputDimensions(imdim_, filters_);
  size_t output_width = std::get<0>(output_dims);
  size_t output_height = std::get<1>(output_dims);
  size_t output_depth = std::get<2>(output_dims);

  symbolic::Expression filter = generator_.GetWeightFilter(index);
  symbolic::Expression weight_x = generator_.GetWeightX(index);
  symbolic::Expression weight_y = generator_.GetWeightY(index);
  symbolic::Expression weight_z = generator_.GetWeightZ(index);

  symbolic::Expression gradients(0.0);
  for (size_t output_x = 0; output_x < output_width; ++output_x) {
    for (size_t output_y = 0; output_y < output_height; ++output_y) {
      symbolic::Expression output_flat_index =
          symbolic::Flatten3d(output_width, output_height, output_depth,
                              output_x, output_y, filter);
      symbolic::Expression input_x =
          (output_x * filters_.stride) - filters_.padding + filters_.width / 2;
      symbolic::Expression input_y =
          (output_y * filters_.stride) - filters_.padding + filters_.height / 2;
      gradients += generator_.GRADIENT(output_flat_index) *
                   generator_.BoundsCheckedI(
                       input_y + weight_y - filters_.height / 2,
                       input_x + weight_x - filters_.width / 2, weight_z);
    }
  }

  cg->AppendLineOfCode("return " + gradients.to_string() + cg->linesep());
}

std::unique_ptr<LayerImpl> ConvolutionLayer::Clone() const {
  return std::make_unique<ConvolutionLayer>(imdim_, filters_,
                                            Super::layer_index_);
}

}  // namespace nnet

#include "math/nnet/convolution_layer.h"

#include <cassert>

namespace nnet {

ConvolutionLayer::ConvolutionLayer(const VolumeDimensions &dimensions,
                                   const FilterParams &filters,
                                   size_t layer_index)
    : Super(GenLinearDimensions(dimensions, filters), layer_index),
      generator_(dimensions, filters), filters_(filters), imdim_(dimensions) {
  if (filters_.depth != imdim_.depth) {
    std::cerr << "Convolution layer input depth != filter depth. Error!"
              << std::endl;
    std::exit(1);
  }
}

LinearDimensions
ConvolutionLayer::GenLinearDimensions(const VolumeDimensions &dim,
                                      const FilterParams &filters) {
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

std::tuple<size_t, size_t, size_t>
ConvolutionLayer::GetOutputDimensions(const VolumeDimensions &dim,
                                      const FilterParams &filters) {
  size_t output_width =
      (dim.width - filters.width + filters.padding * 2) / filters.stride + 1;
  size_t output_height =
      (dim.height - filters.height + filters.padding * 2) / filters.stride + 1;
  size_t output_depth = filters.num_filters;
  return std::make_tuple(output_width, output_height, output_depth);
}

const std::vector<std::string> &ConvolutionLayer::weights() const {
  return generator_.weights();
}

void ConvolutionLayer::GenerateOutputCode(const symbolic::Expression &index,
                                          codegen::Generator *cg) const {
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

  // Sum up the convolution, adding it to the output.
  cg->AppendLineOfCode(
      cg->assign("double output", generator_.W(output_filter).to_string()) +
      cg->linesep());
  symbolic::Expression f_x = symbolic::Expression::CreateInteger("f_x");
  symbolic::Expression f_y = symbolic::Expression::CreateInteger("f_y");
  symbolic::Expression f_z = symbolic::Expression::CreateInteger("f_z");
  symbolic::Expression input_x = conv_start_row + f_x - filters_.width/2;
  symbolic::Expression input_y = conv_start_col + f_y - filters_.height/2;
  symbolic::Expression input_z = f_z;
  symbolic::Expression output_factor =
      generator_.W(output_filter, f_x, f_y, f_z) *
      generator_.BoundsCheckedI(input_x, input_y, input_z);
  string output_sum =
      cg->add_assign("output", output_factor.to_string() + cg->linesep());
  string for_loop_z =
      cg->for_loop("size_t f_z = 0", "f_z < " + std::to_string(filters_.depth),
                   "++f_z", output_sum);
  string for_loop_yz =
      cg->for_loop("size_t f_y = 0", "f_y < " + std::to_string(filters_.height),
                   "++f_y", for_loop_z);
  string for_loop_xyz =
      cg->for_loop("size_t f_x = 0", "f_x < " + std::to_string(filters_.width),
                   "++f_x", for_loop_yz);
  cg->AppendLineOfCode(for_loop_xyz);
  cg->AppendLineOfCode("return output" + cg->linesep());
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
  // Sum up the convolution, adding it to the output.
  cg->AppendLineOfCode(cg->assign("double gradient", "0") + cg->linesep());
  symbolic::Expression d = symbolic::Expression::CreateInteger("d");
  symbolic::Expression k = symbolic::Expression::CreateInteger("k");
  symbolic::Expression filter = symbolic::Expression::CreateInteger("filter");
  symbolic::Expression z = symbolic::Expression::CreateInteger("z");
  symbolic::Expression neighbor_output_flat_index =
      symbolic::Flatten3d(output_width, output_height, output_depth,
                          output_row + d, output_col + k, filter);
  symbolic::Expression self_in_neighbor_row = (d * -1) + output_net_width / 2;
  symbolic::Expression self_in_neighbor_col = (k * -1) + output_net_height / 2;
  // When looking at the gradient component propagated from a neighbor
  // output, we want to consider which weight this input is multiplied by
  // to generate that output (this is the derivative wrt ourselves). So to
  // get this, we need to figure out the relative coordinate of this input
  // within the convolution domain of the neighboring output. Thinking of
  // the 1-dimensional case, say you have a convolution of width 3. If you
  // are looking for the backprop gradients of position I in an input
  // array...
  // Example 1D convolution:
  //                 [ w1, w2, w3]
  // [i[-3], i[-2], i[-1], i, i[+1], i[+2], i[+3]]
  //
  // output = (i[-1] * w1) + (i * w2) + (i[+1] * w3)
  //
  // derivative of output w.r.t. i[-1] = w1.
  //
  // So for all partial derivatives wrt to some input, we need to find the
  // weights (w1 in this case) multiplied by this input in order to
  // generate all neighboring outputs.
  symbolic::Expression gradient_factor =
      generator_.W(filter, self_in_neighbor_row, self_in_neighbor_col, z) *
      generator_.GRADIENT(neighbor_output_flat_index);
  string output_sum =
      cg->add_assign("gradient", gradient_factor.to_string() + cg->linesep());
  string for_loop_z =
      cg->for_loop("size_t z = 0", "z < " + std::to_string(filters_.depth),
                   "++z", output_sum);
  string for_loop_fz = cg->for_loop(
      "size_t filter = 0", "filter < " + std::to_string(filters_.num_filters),
      "++filter", for_loop_z);
  string for_loop_kfz = cg->for_loop(
      "size_t k = " + std::to_string(-output_net_height / 2),
      "k <= " + std::to_string(output_net_height / 2), "++k", for_loop_fz);
  string for_loop_dkfz = cg->for_loop(
      "size_t d = " + std::to_string(-output_net_width / 2),
      "d <= " + std::to_string(output_net_width / 2), "++d", for_loop_kfz);
  cg->AppendLineOfCode(for_loop_dkfz);
  cg->AppendLineOfCode("return gradient" + cg->linesep());
}

void ConvolutionLayer::WeightGradientCode(const symbolic::Expression &index,
                                          codegen::Generator *cg) const {
  std::tuple<size_t, size_t, size_t> output_dims =
      GetOutputDimensions(imdim_, filters_);
  size_t output_width = std::get<0>(output_dims);
  size_t output_height = std::get<1>(output_dims);
  size_t output_depth = std::get<2>(output_dims);

  symbolic::Expression filter = generator_.GetWeightFilter(index);
  symbolic::Expression weight_x = generator_.GetWeightX(index);
  symbolic::Expression weight_y = generator_.GetWeightY(index);
  symbolic::Expression weight_z = generator_.GetWeightZ(index);

  cg->AppendLineOfCode(cg->assign("double gradient", "0") + cg->linesep());
  symbolic::Expression out_x = symbolic::Expression::CreateInteger("out_x");
  symbolic::Expression out_y = symbolic::Expression::CreateInteger("out_y");
  symbolic::Expression output_flat_index = symbolic::Flatten3d(
      output_width, output_height, output_depth, out_x, out_y, filter);
  symbolic::Expression input_x =
      (out_x * filters_.stride) - filters_.padding + filters_.width / 2;
  symbolic::Expression input_y =
      (out_y * filters_.stride) - filters_.padding + filters_.height / 2;
  symbolic::Expression gradient_factor =
      generator_.GRADIENT(output_flat_index) *
      generator_.BoundsCheckedI(input_y + weight_y - filters_.height / 2,
                                input_x + weight_x - filters_.width / 2,
                                weight_z);
  string output_sum =
      cg->add_assign("gradient", gradient_factor.to_string() + cg->linesep());
  string for_loop_y = cg->for_loop("size_t out_y = 0",
                                   "out_y < " + std::to_string(output_width),
                                   "++out_y", output_sum);
  string for_loop_xy = cg->for_loop("size_t out_x = 0",
                                    "out_x < " + std::to_string(output_height),
                                    "++out_x", for_loop_y);
  cg->AppendLineOfCode(for_loop_xy);
  cg->AppendLineOfCode("return gradient" + cg->linesep());
}

std::unique_ptr<LayerImpl> ConvolutionLayer::Clone() const {
  return std::make_unique<ConvolutionLayer>(imdim_, filters_,
                                            Super::layer_index_);
}

} // namespace nnet

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

  symbolic::Expression conv_row, conv_col;
  std::tie(conv_row, conv_col) = GetInputCoordinates(output_row, output_col);

  // Sum up the convolution, adding it to the output.
  cg->AppendLineOfCode(
      cg->assign("double output", generator_.W(output_filter).to_string()) +
      cg->linesep());
  symbolic::Expression f_x = symbolic::Expression::CreateInteger("f_x");
  symbolic::Expression f_y = symbolic::Expression::CreateInteger("f_y");
  symbolic::Expression f_z = symbolic::Expression::CreateInteger("f_z");
  symbolic::Expression input_x = conv_col + f_x - filters_.width/2;
  symbolic::Expression input_y = conv_row + f_y - filters_.height/2;
  symbolic::Expression input_z = f_z;
  symbolic::Expression output_factor =
      generator_.W(output_filter, f_y, f_x, f_z) *
      generator_.BoundsCheckedI(input_y, input_x, input_z);
  string output_sum =
      cg->add_assign("output", output_factor.to_string() + cg->linesep());
  string for_loop_z =
      cg->for_loop("int f_z = 0", "f_z < " + std::to_string(filters_.depth),
                   "++f_z", output_sum);
  string for_loop_yz =
      cg->for_loop("int f_y = 0", "f_y < " + std::to_string(filters_.height),
                   "++f_y", for_loop_z);
  string for_loop_xyz =
      cg->for_loop("int f_x = 0", "f_x < " + std::to_string(filters_.width),
                   "++f_x", for_loop_yz);
  cg->AppendLineOfCode(for_loop_xyz);
  cg->AppendLineOfCode("return output" + cg->linesep());
}

// Returns Row, Col symbolic.
std::tuple<symbolic::Expression, symbolic::Expression>
ConvolutionLayer::GetInputCoordinates(
    const symbolic::Expression &output_row,
    const symbolic::Expression &output_col) const {
  return std::make_tuple<symbolic::Expression, symbolic::Expression>(
      output_row * filters_.stride - filters_.padding + (filters_.height / 2),
      output_col * filters_.stride - filters_.padding + (filters_.width / 2));
}

// Returns Row, Col symbolic.
std::tuple<symbolic::Expression, symbolic::Expression>
ConvolutionLayer::GetOutputCoordinates(
    const symbolic::Expression &input_row,
    const symbolic::Expression &input_col) const {
  return std::make_tuple<symbolic::Expression, symbolic::Expression>(
      (input_row + filters_.padding - (filters_.height / 2)) / filters_.stride,
      (input_col + filters_.padding - (filters_.width / 2)) / filters_.stride);
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

  symbolic::Expression output_row, output_col;
  std::tie(output_row, output_col) = GetOutputCoordinates(input_row, input_col);

  // The "net" includes all outputs which depend on this input. This is
  // determined in the input coordinate domain by the filter size. After finding
  // the farthest points in the input domain for which the convolution filter
  // would include this input, those farthest points are converted to the output
  // domain to create the output net.
  int input_net_width = filters_.width / 2;
  int input_net_height = filters_.height / 2;

  // The two farthest points are referred to as "a" and "b" (a being closer to
  // (0, 0) and b being closer to (width, height)). These are opposite corners
  // of a rectangle containing all points which, if used as the center of a
  // convolution, would include the currently selected input.
  symbolic::Expression input_a_row = input_row - input_net_height;
  symbolic::Expression input_a_col = input_col - input_net_width;
  symbolic::Expression input_b_row = input_row + input_net_height;
  symbolic::Expression input_b_col = input_col + input_net_width;
  
  // Convert the net endpoints from the input domain to the output domain.
  // This may result in the rectangle not perfectly covering the same points, as
  // input->output is a many to one relationship. However, in the case where a
  // point is covered which does not include this input in its convolution,
  // bounds checking will result in the gradient for that point to be zero (see
  // BoundsCheckedW()) and in the case where a point is not covered, it is
  // because there exists no output centered on that input point (in cases where
  // stride > 1), so rounding down will give us the furthest point in that
  // direction which might cover this input in its convolution.
  symbolic::Expression output_a_row, output_a_col, output_b_row, output_b_col;
  std::tie(output_a_row, output_a_col) = GetOutputCoordinates(input_a_row, input_a_col);
  std::tie(output_b_row, output_b_col) = GetOutputCoordinates(input_b_row, input_b_col);

  // Sum up the convolution, adding it to the output.
  cg->AppendLineOfCode(cg->assign("double gradient", "0") + cg->linesep());
  // d iterates from output_a_row to output_b_row.
  symbolic::Expression d = symbolic::Expression::CreateInteger("d");
  // k iterates from output_a_col to output_b_col.
  symbolic::Expression k = symbolic::Expression::CreateInteger("k");
  symbolic::Expression filter = symbolic::Expression::CreateInteger("filter");
  symbolic::Expression z = input_plane;
  symbolic::Expression neighbor_output_flat_index =
      symbolic::Flatten3d(output_width, output_height, output_depth,
                          d, k, filter);

  symbolic::Expression input_d, input_k;
  std::tie(input_d, input_k) = GetInputCoordinates(d, k);
  symbolic::Expression self_in_neighbor_row = input_row - input_d + (filters_.height / 2);
  symbolic::Expression self_in_neighbor_col = input_col - input_k + (filters_.width / 2);
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
      generator_.BoundsCheckedW(filter, self_in_neighbor_row, self_in_neighbor_col, z) *
      generator_.GRADIENT(neighbor_output_flat_index);
  symbolic::Expression bounds_checked_gradient_factor = IfInRange(input_d, 0, imdim_.height, IfInRange(input_k, 0, imdim_.width, gradient_factor, 0.0), 0.0);
  string output_sum = cg->add_assign(
      "gradient", bounds_checked_gradient_factor.to_string() + cg->linesep());

  string for_loop_f = cg->for_loop(
      "int filter = 0", "filter < " + std::to_string(filters_.num_filters),
      "++filter", output_sum);
  string for_loop_kf =
      cg->for_loop("int k = " + output_a_col.to_string(),
                   "k <= " + output_b_col.to_string(), "++k", for_loop_f);
  string for_loop_dkf =
      cg->for_loop("int d = " + output_a_row.to_string(),
                   "d <= " + output_b_row.to_string(), "++d", for_loop_kf);
  cg->AppendLineOfCode(for_loop_dkf);
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
      output_width, output_height, output_depth, out_y, out_x, filter);
  symbolic::Expression input_y, input_x;
  std::tie(input_y, input_x) = GetInputCoordinates(out_y, out_x);
  // Correctly handle the input for bias weight indices.
  size_t filter_size = filters_.width * filters_.height * filters_.depth + 1;
  symbolic::Expression input = symbolic::IfInRange(
      index % filter_size, filter_size - 1, filter_size, 1.0,
      // input_y is the center of the convolution, but weight coordinates use
      // (0, 0) as the top-left, so subtract filters_.(height or width)/2 to
      // translate coordinates.
      generator_.BoundsCheckedI(input_y + weight_y - (filters_.height / 2),
                                input_x + weight_x - (filters_.width / 2),
                                weight_z));
  symbolic::Expression gradient_factor =
      generator_.GRADIENT(output_flat_index) * input;
  string output_sum =
      cg->add_assign("gradient", gradient_factor.to_string() + cg->linesep());
  string for_loop_y = cg->for_loop("int out_y = 0",
                                   "out_y < " + std::to_string(output_width),
                                   "++out_y", output_sum);
  string for_loop_xy = cg->for_loop("int out_x = 0",
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

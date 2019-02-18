#include "math/nnet/max_pool_layer.h"
#include "math/symbolic/symbolic_util.h"

namespace nnet {

MaxPoolLayer::MaxPoolLayer(const VolumeDimensions& input,
                           const AreaDimensions& output, size_t layer_index)
    : Super(MaxPoolLayer::GenLinearDimensions(input, output), layer_index),
      generator_(input),
      input_(input),
      target_(output) {
  if (dimensions_.num_outputs > dimensions_.num_inputs) {
    std::cerr << "Error: MaxPoolLayer " << layer_index
              << " constructed with more outputs than inputs?" << std::endl
              << "3D inputs: " << input.width << ", " << input.height << ", "
              << input.depth << std::endl
              << "3D outputs: " << output.width << ", " << output.height << ", "
              << input.depth << std::endl
              << "Numbler inputs: " << dimensions_.num_inputs << std::endl
              << "Number outputs: " << dimensions_.num_outputs << std::endl;
    std::exit(1);
  }
  // The input dimensions must be multiples of the output dimensions. This is
  // pooling pixels, not resizing, all pooling groups must be the same size with
  // no remainer.
  if ((input.width % output.width != 0) ||
      (input.height % output.height != 0)) {
    std::cerr << "MaxPool layer #" << layer_index
              << " specified with output height or width which is not an even "
                 "divisor of the input dimensions"
              << std::endl;
  }
}

std::tuple<size_t, size_t, size_t> MaxPoolLayer::GetOutputDimensions(
    const VolumeDimensions& dim, const AreaDimensions& output) {
  return std::make_tuple(output.width, output.height, dim.depth);
}

void MaxPoolLayer::GenerateOutputCode(const symbolic::Expression &index,
                                      codegen::Generator *cg) const {
  // Get 3D output dimensions.
  std::tuple<size_t, size_t, size_t> output_dims =
      GetOutputDimensions(input_, target_);
  size_t output_width = std::get<0>(output_dims);
  size_t output_height = std::get<1>(output_dims);
  size_t output_depth = std::get<2>(output_dims);

  size_t group_width = input_.width / output_width;
  size_t group_height = input_.height / output_height;

  symbolic::Expression output_row = symbolic::Unflatten3dRow(
      output_width, output_height, output_depth, index);

  symbolic::Expression output_col = symbolic::Unflatten3dCol(
      output_width, output_height, output_depth, index);

  symbolic::Expression output_z = symbolic::Unflatten3dPlane(
      output_width, output_height, output_depth, index);

  symbolic::Expression group_r_start = output_row * group_height;
  symbolic::Expression group_c_start = output_col * group_width;

  symbolic::Expression group_r = symbolic::Expression::CreateInteger("group_r");
  symbolic::Expression group_c = symbolic::Expression::CreateInteger("group_c");
  symbolic::Expression depth = symbolic::Expression::CreateInteger("depth");
  symbolic::Expression max = symbolic::Expression::CreateNumericValue("max");
  cg->AppendLineOfCode("float " + max.to_string() + " = -INFINITY" + cg->linesep());
  cg->AppendLineOfCode("for (size_t group_r = 0; group_r < " +
                       std::to_string(group_height) + "; ++group_r)");
  cg->PushScope();
  cg->AppendLineOfCode("for (size_t group_c = 0; group_c < " +
                       std::to_string(group_width) + "; ++group_c)");
  cg->PushScope();
  symbolic::Expression current_input = generator_.I(group_r_start + group_r, group_c_start + group_c, output_z);
  cg->AppendLineOfCode(
      cg->if_expr(cg->gt(current_input.to_string(), max.to_string())));
  cg->PushScope();
  cg->AppendLineOfCode(cg->assign(max.to_string(), current_input.to_string()) + cg->linesep()); 
  cg->PopScope();
  cg->PopScope();
  cg->PopScope();
  cg->AppendLineOfCode("return " + max.to_string() + cg->linesep());
}

void MaxPoolLayer::InputGradientCode(
    const symbolic::Expression& input_index, codegen::Generator *cg) const {
  // Okay so for this input, find the group it belongs to.
  std::tuple<size_t, size_t, size_t> output_dims =
      GetOutputDimensions(input_, target_);
  size_t output_width = std::get<0>(output_dims);
  size_t output_height = std::get<1>(output_dims);
  size_t output_depth = std::get<2>(output_dims);
  size_t input_depth = output_depth;

  size_t group_width = input_.width / output_width;
  size_t group_height = input_.height / output_height;

  symbolic::Expression input_row = symbolic::Unflatten3dRow(
      input_.width, input_.height, input_depth, input_index);

  symbolic::Expression input_col = symbolic::Unflatten3dCol(
      input_.width, input_.height, input_depth, input_index);

  symbolic::Expression input_z = symbolic::Unflatten3dPlane(
      input_.width, input_.height, input_depth, input_index);

  symbolic::Expression output_row = input_row / group_height;
  symbolic::Expression output_col = input_col / group_width;
  symbolic::Expression output_z = input_z;
  symbolic::Expression output_flat_index =
      symbolic::Flatten3d(output_width, output_height, output_depth, output_row,
                          output_col, output_z);

  symbolic::Expression group_r_start = output_row * group_height;
  symbolic::Expression group_c_start = output_col * group_width;

  symbolic::Expression group_r = symbolic::Expression::CreateInteger("group_r");
  symbolic::Expression group_c = symbolic::Expression::CreateInteger("group_c");
  symbolic::Expression depth = symbolic::Expression::CreateInteger("depth");
  symbolic::Expression current_input = generator_.I(input_row, input_col, input_z);
  cg->AppendLineOfCode("for (size_t group_r = 0; group_r < " +
                       std::to_string(group_height) + "; ++group_r)");
  cg->PushScope();
  cg->AppendLineOfCode("for (size_t group_c = 0; group_c < " +
                       std::to_string(group_width) + "; ++group_c)");
  cg->PushScope();
  symbolic::Expression value = generator_.I(group_r_start + group_r, group_c_start + group_c, output_z);
  cg->AppendLineOfCode(
      cg->if_expr(cg->gt(value.to_string(), current_input.to_string())));
  cg->PushScope();
  cg->AppendLineOfCode("return 0.0" + cg->linesep()); 
  cg->PopScope();
  cg->PopScope();
  cg->PopScope();
  cg->AppendLineOfCode("return " +
                       generator_.GRADIENT(output_flat_index).to_string() +
                       cg->linesep());
}

void MaxPoolLayer::WeightGradientCode(
    const symbolic::Expression& weight_index, codegen::Generator *cg) const {
  cg->AppendLineOfCode("return 0.0" + cg->linesep());
}

std::unique_ptr<LayerImpl> MaxPoolLayer::Clone() const {
  return std::make_unique<MaxPoolLayer>(input_, target_, layer_index_);
}

}  // namespace nnet

#include "math/nnet/max_pool_layer.h"
#include "math/symbolic/symbolic_util.h"

namespace nnet {

MaxPoolLayer::MaxPoolLayer(const VolumeDimensions& input,
                           const AreaDimensions& output,
                           SymbolGenerator* generator, size_t layer_index)
    : Super(MaxPoolLayer::GenLinearDimensions(input, output), generator,
            layer_index),
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

Matrix<symbolic::Expression> MaxPoolLayer::GenerateExpression(
    const Matrix<symbolic::Expression>& input) {
  auto dim = input.size();
  size_t rows = std::get<0>(dim);
  size_t cols = std::get<1>(dim);
  if ((rows != dimensions_.num_inputs) || (cols != 1)) {
    std::cerr << "Error: MaxPoolLayer::GenerateExpression called on input "
                 "of incorrect size: "
              << "(" << rows << ", " << cols << ")" << std::endl;
    std::exit(1);
  }

  // Get 3D output dimensions. (output will be a 1D serialized form of this,
  // using mapping output_flat_index).
  std::tuple<size_t, size_t, size_t> output_dims =
      GetOutputDimensions(input_, target_);
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
    size_t row_index = y * (input_.width * input_.depth);
    size_t col_index = x * input_.depth;
    size_t depth_index = z;
    return row_index + col_index + depth_index;
  };

  std::function<size_t(size_t, size_t, size_t)> output_flat_index =
      [output_width, output_height, output_depth](size_t x, size_t y,
                                                  size_t z) -> size_t {
    size_t row_index = y * (output_width * output_depth);
    size_t col_index = x * output_depth;
    size_t depth_index = z;
    return row_index + col_index + depth_index;
  };

  Matrix<symbolic::Expression> output(
      output_width * output_height * output_depth, 1);

  size_t group_width = input_.width / output_width;
  size_t group_height = input_.height / output_height;

  for (size_t out_r = 0; out_r < output_height; ++out_r) {
    for (size_t out_c = 0; out_c < output_width; ++out_c) {
      for (size_t out_d = 0; out_d < output_depth; ++out_d) {
        std::vector<symbolic::Expression> group;
        size_t group_r_start = out_r * group_height;
        size_t group_r_end = (out_r + 1) * group_height;
        size_t group_c_start = out_c * group_width;
        size_t group_c_end = (out_c + 1) * group_width;
        for (size_t group_r = group_r_start; group_r < group_r_end; ++group_r) {
          for (size_t group_c = group_c_start; group_c < group_c_end;
               ++group_c) {
            group.push_back(
                input.at(input_flat_index(group_r, group_c, out_d), 0));
          }
        }
        symbolic::Expression group_max = symbolic::Max(group);
        output.at(output_flat_index(out_r, out_c, out_d), 0) = group_max;
      }
    }
  }

  return output;
}

std::unique_ptr<LayerImpl> MaxPoolLayer::Clone() const {
  return std::make_unique<MaxPoolLayer>(input_, target_, generator_,
                                        layer_index_);
}

}  // namespace nnet

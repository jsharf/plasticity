#include "math/geometry/dynamic_matrix.h"
#include "math/nnet/layer.h"
#include "math/stats/normal.h"
#include "math/symbolic/expression.h"
#include "math/symbolic/symbolic_util.h"

#include <tuple>

namespace nnet {

// Parameter sharing is enforced. Each filter has its own weights, but the
// weights are the same regardless of which part of the input is convolved with
// the filter.
class ConvolutionLayer : public LayerGenerator {
 public:
  // Reference objects in superclass with Super::
  using Super = LayerGenerator;
  using WeightArray = typename Super::WeightArray;
  using LinearDimensions = typename Super::Dimensions;

  struct FilterParams {
    // Dimensions of each filter.
    size_t width;
    size_t height;
    size_t depth;

    // Filter stride. PS If not sure, set to 1.
    size_t stride;

    // Zero-padding on input image. This is the number of zeroes (pixels) added
    // to *each* side of the input image when doing a convolution with the
    // filter.
    size_t padding;

    // The number of filters.
    size_t num_filters;
  };

  // Really just a 3D volume, but first layer is a WxHx3 image.
  struct ImageDimensions {
    size_t width;
    size_t height;

    // 1 for grey, 3 for rgb. Or whatever, it's really just an input volume,
    // this is a convolutional layer. The first layer takes an image as an
    // input, so this is named ImageDimensions. Sorry if that's confusing,
    // that's why I left this comment.
    size_t depth;
  };

  static LinearDimensions GenLinearDimensions(const ImageDimensions& dim,
                                              const FilterParams& params) {
    std::tuple<size_t, size_t, size_t> output_dims =
        GetOutputDimensions(dim, params);
    size_t output_width = std::get<0>(output_dims);
    size_t output_height = std::get<1>(output_height);
    size_t output_depth = std::get<2>(output_depth);
    return LinearDimensions{
        // Num Inputs.
        dim.width * dim.height * dim.depth,
        // Num Outputs.
        output_width * output_height * output_depth,
    };
  }

  // Returns output volume dim (width, height, depth).
  static std::tuple<size_t, size_t, size_t> GetOutputDimensions(
      const ImageDimensions& dim, const FilterParams& params) {
    size_t output_width =
        (dim.width - params.width + params.padding * 2) / params.stride + 1;
    size_t output_height =
        (dim.height - params.height + params.padding * 2) / params.stride + 1;
    size_t output_depth = params.num_filters;
    return std::make_tuple(output_width, output_height, output_depth);
  }

  ConvolutionLayer(const ImageDimensions& dimensions,
                   const FilterParams& params, SymbolGenerator* generator,
                   size_t layer_index)
      : Super(GenLinearDimensions(dimensions, params), generator, layer_index),
        imdim_(dimensions),
        filters_(params) {
    if (filters_.depth != imdim_.depth) {
      std::cerr << "Convolution layer input depth != filter depth. Error!"
                << std::endl;
      std::exit(1);
    }
  }

  WeightArray weights() override {
    WeightArray weights(
        filters_.num_filters *
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

  Matrix<symbolic::Expression> GenerateExpression(
      Matrix<symbolic::Expression> input) override {
    auto dim = input.size();
    size_t rows = std::get<0>(dim);
    size_t cols = std::get<1>(dim);

    // Validate input dimensions.
    if ((rows != dimensions_.num_inputs) || (cols != 1)) {
      std::cerr
          << "Error: ConvolutionLayer::GenerateExpression called on input "
             "of incorrect size: "
          << "(" << rows << ", " << cols << ")" << std::endl;
      std::exit(1);
    }

    // Get 3D output dimensions. (output will be a 1D serialized form of this,
    // using mapping output_flat_index).
    std::tuple<size_t, size_t, size_t> output_dims =
        GetOutputDimensions(dim, params);
    size_t output_width = std::get<0>(output_dims);
    size_t output_height = std::get<1>(output_height);
    size_t output_depth = std::get<2>(output_depth);

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
    }

    for (size_t filter_no = 0; filter_no < filters_.num_filters; ++filter_no) {
      size_t start_x = -params.padding;
      size_t start_y = -params.padding;
      for (size_t out_x = 0; out_x < output_width; ++out_x) {
        for (size_t out_y = 0; out_y < output_height; ++out_y) {
          symbolic::Expression convolution = symbolic::CreateExpression("0");
          for (size_t f_x = 0; f_x < filters_.width; ++f_x) {
            for (size_t f_y = 0; f_y < filters_.height; ++f_y) {
              for (size_t f_z = 0; f_z < filters_.depth; ++f_z) {
                size_t input_x = start_x + (out_x * filters_.stride) + f_x;
                size_t input_y = start_y + (out_y * filters_.stride) + f_y;
                size_t input_z = f_z;
                // Make sure that we aren't indexing out-of-bounds for
                // zero-padding case.
                if ((input_x >= 0) && (input_x < imdim_.width) &&
                    (input_y >= 0) && (input_y < imdim_.height) &&
                    (input_z >= 0) && (input_z < imdim_.depth)) {
                  convolution +=
                      input.at(input_flat_index(input_x, input_y, input_z)) *
                      symbolic::CreateExpression(Super::generator_->W(
                          Super::layer_index_, filter_no, f_x, f_y, f_z));
                }
              }
            }
          }
          output_matrix.at(output_flat_index(out_x, out_y, filter_no), 1) =
              convolution;
        }
      }
    }

    return output_matrix;
  }

  stats::Normal XavierInitializer() const override {
    // + filters_.num_filters since each filter has an implicit bias input.
    return stats::Normal(0,
                         1.0 / (dimensions_.num_inputs + filters_.num_filters));
  }

  FilterParams filters_;
  ImageDimensions imdim_;
};
}  // namespace nnet

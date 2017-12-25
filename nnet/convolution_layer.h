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
    std::tuple<size_t, size_t, size_t> output_dims = GetOutputDimensions(dim, params);
    size_t output_width = std::get<0>(output_dims);
    size_t output_height = std::get<1>(output_height);
    size_t output_depth = std::get<2>(output_depth);
    return LinearDimensions {
      // Num Inputs.
      dim.width * dim.height * dim.depth,
      // Num Outputs.
      output_width * output_height * output_depth,
    }
  }

  // Returns output volume dim (width, height, depth).
  static std::tuple<size_t, size_t, size_t> GetOutputDimensions(const ImageDimensions& dim, const FilterParams& params) {
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
            std::cerr << "Convolution layer input depth != filter depth. Error!" << std::endl;
            std::exit(1);
          }
        }

  WeightArray weights() override {
    WeightArray weights(filters_.num_filters * (filters_.width * filters_.height * filters_.depth + 1));
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
      weights[back_index++] = Super::generator_->W(Super::layer_index_, filter_no);
    }
    return weights;
  }

  Matrix<symbolic::Expression> GenerateExpression(Matrix<symbolic::Expression> input) override {
    auto dim = input.size();
    size_t rows = std::get<0>(dim);
    size_t cols = std::get<1>(dim);
    if ((rows != dimensions_.num_inputs) || (cols != 1)) {
      std::cerr << "Error: ConvolutionLayer::GenerateExpression called on input "
                   "of incorrect size: "
                << "(" << rows << ", " << cols << ")" << std::endl;
      std::exit(1);
    }

    for (size_t filter_no = 0; filter_no < filters_.num_filters; ++filter_no) {
      for (size_t z = 0; z < imdim_.depth; ++z) {
        size_t start_x = filters_.width/2 - filters_.padding;
        size_t end_x = (imdim_.width - 1) - filters_.width/2
        for (size_t x = start_x; x <= end_x; ++x) {
        Matrix<symbolic::Expression> convolution_matrix(dimensions_.num_outputs,
                                                   dimensions_.num_inputs + 1);
      }
    }

    return (weight_matrix * biased_input).Map(activation_function_);
  }

  stats::Normal XavierInitializer() const override {
    // + filters_.num_filters since each filter has an implicit bias input.
    return stats::Normal(0, 1.0 / (dimensions_.num_inputs + filters_.num_filters));
  }

  FilterParams filters_;
  ImageDimensions imdim_;

};
}  // namespace nnet

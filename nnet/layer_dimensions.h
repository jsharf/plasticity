#ifndef LAYER_DIMENSIONS_H
#define LAYER_DIMENSIONS_H
namespace nnet {
  struct Dimensions {
    size_t num_inputs;
    size_t num_outputs;
  };

  // An alias to make life easier for classes which refer to a layer by more
  // than one dimension type.
  using LinearDimensions=Dimensions;

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

  struct VolumeDimensions {
    size_t width;
    size_t height;

    // 1 for grey, 3 for rgb. Or whatever, it's really just an input volume,
    // this is a convolutional layer.
    size_t depth;
  };

  struct AreaDimensions {
    size_t width, height;
  };

} // namespace nnet
#endif // LAYER_DIMENSIONS_H

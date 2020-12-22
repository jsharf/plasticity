#include "nnet/yolov1.h"
#include "nnet/convolution_layer.h"

namespace nnet {

namespace {
  Architecture YoloArchitecture() {
    nnet::Architecture model(kInputSize);
    model
        .AddConvolutionLayer(
            {
                448,  // width
                448,  // height
                3,   // R,G,B (depth).
            },
            {
                7,   // filter x size.
                7,   // filter y size.
                3,   // filter z depth size.
                2,   // stride.
                3,   // padding.
                64,  // number of filters.
            })
        .AddMaxPoolLayer(
            /* Input size */ nnet::VolumeDimensions{224, 224, 64},
            /* Output size */ nnet::AreaDimensions{112, 112})
        .AddConvolutionLayer(
            {
                112,  // width
                112,  // height
                64,  // R,G,B (depth).
            },
            {
                3,   // filter x size.
                3,   // filter y size.
                64,  // filter z depth size.
                1,   // stride.
                2,   // padding.
                192,  // number of filters.
            })
        .AddMaxPoolLayer(
            /* Input size */ nnet::VolumeDimensions{112, 112, 192},
            /* output size */ nnet::AreaDimensions{56, 56})
        .AddConvolutionLayer(
            {
                56,   // width
                56,   // height
                192,  // R,G,B (depth).
            },
            {
                1,   // filter x size.
                1,   // filter y size.
                192,  // filter z depth size.
                1,   // stride.
                0,   // padding.
                128,  // number of filters.
            })
        .AddConvolutionLayer(
            {
                56,   // width
                56,   // height
                128,  // R,G,B (depth).
            },
            {
                3,   // filter x size.
                3,   // filter y size.
                128,  // filter z depth size.
                1,   // stride.
                1,   // padding.
                256,  // number of filters.
            })
        .AddConvolutionLayer(
            {
                56,   // width
                56,   // height
                256,  // R,G,B (depth).
            },
            {
                1,   // filter x size.
                1,   // filter y size.
                256,  // filter z depth size.
                1,   // stride.
                0,   // padding.
                256,  // number of filters.
            })
        .AddConvolutionLayer(
            {
                56,   // width
                56,   // height
                256,  // R,G,B (depth).
            },
            {
                3,   // filter x size.
                3,   // filter y size.
                256,  // filter z depth size.
                1,   // stride.
                1,   // padding.
                512,  // number of filters.
            })
        .AddMaxPoolLayer(/* Input size */ {56, 56, 512},
                         /* output size */ {28, 28});
        for (size_t i = 0; i < 4; ++i) {
          model.AddConvolutionLayer(
            {
                28,   // width
                28,   // height
                512,  // R,G,B (depth).
            },
            {
                1,   // filter x size.
                1,   // filter y size.
                512,  // filter z depth size.
                1,   // stride.
                0,   // padding.
                256,  // number of filters.
            }).AddConvolutionLayer(
            {
                28,   // width
                28,   // height
                256,  // R,G,B (depth).
            },
            {
                3,   // filter x size.
                3,   // filter y size.
                256,  // filter z depth size.
                1,   // stride.
                1,   // padding.
                512,  // number of filters.
            });
        }
        model.AddConvolutionLayer({
            28,   // width
            28,   // height
            512,  // R,G,B (depth).
        }, {
            1,   // filter x size.
            1,   // filter y size.
            512,  // filter z depth size.
            1,   // stride.
            0,   // padding.
            512,  // number of filters.
        })
        .AddMaxPoolLayer(/* Input size */ {28, 28, 512},
                         /* output size */ {14, 14});
        for (size_t i = 0; i < 2; ++i) {
          model.AddConvolutionLayer(
            {
                14,   // width
                14,   // height
                512,  // R,G,B (depth).
            },
            {
                1,   // filter x size.
                1,   // filter y size.
                512,  // filter z depth size.
                1,   // stride.
                0,   // padding.
                512,  // number of filters.
            }).AddConvolutionLayer(
            {
                14,   // width
                14,   // height
                512,  // R,G,B (depth).
            },
            {
                3,   // filter x size.
                3,   // filter y size.
                512,  // filter z depth size.
                1,   // stride.
                1,   // padding.
                1024,  // number of filters.
            });
        }
        model.AddConvolutionLayer(
            {
                14,   // width
                14,   // height
                1024,  // R,G,B (depth).
            },
            {
                3,   // filter x size.
                3,   // filter y size.
                1024,  // filter z depth size.
                1,   // stride.
                1,   // padding.
                1024,  // number of filters.
            })
        .AddConvolutionLayer(
            {
                14,   // width
                14,   // height
                1024,  // R,G,B (depth).
            },
            {
                3,   // filter x size.
                3,   // filter y size.
                1024,  // filter z depth size.
                2,   // stride.
                1,   // padding.
                1024,  // number of filters.
            })
        .AddConvolutionLayer(
            {
                7,   // width
                7,   // height
                1024,  // R,G,B (depth).
            },
            {
                3,   // filter x size.
                3,   // filter y size.
                1024,  // filter z depth size.
                1,   // stride.
                1,   // padding.
                1024,  // number of filters.
            })
        .AddConvolutionLayer(
            {
                7,   // width
                7,   // height
                1024,  // R,G,B (depth).
            },
            {
                3,   // filter x size.
                3,   // filter y size.
                1024,  // filter z depth size.
                1,   // stride.
                1,   // padding.
                1024,  // number of filters.
            })
        .AddDenseLayer(4096)
        .AddDenseLayer(1470);  // 7 * 7 * 30 is the yolov1 output tensor dimension.
        // No activation function, the next layer is softmax which functions as an
        // activation function
        .AddDenseLayer(10, symbolic::Identity)
        .AddSoftmaxLayer(10);
        }
  }
}  // namespace

Yolov1::Yolov1() :
    net_(YoloArchitecture(), nnet::Nnet::Xavier, nnet::CrossEntropy) {
}

}  // namespace nnet

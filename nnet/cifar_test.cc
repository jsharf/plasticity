#include "math/geometry/dynamic_matrix.h"
#include "math/nnet/nnet.h"
#include "math/symbolic/expression.h"

#include <cassert>
#include <fstream>
#include <iostream>
#include <string>

constexpr size_t kSampleSize = 2;
// Dim(kSampleSize, 1)
using Sample = Matrix<nnet::Number>;

// Trains a neural network to learn if given point is in unit circle.
int main() {
  constexpr int kLayerSize = 4;
  constexpr int kOutputSize = 1;
  constexpr int kInputSize = kSampleSize;

  nnet::Nnet::Architecture model(kInputSize);
  model.AddConvolutionLayer(
      {
          32,  // width
          32,  // height
          3,   // R,G,B (depth).
      },
      {
          5,   // filter x size.
          5,   // filter y size.
          3,   // filter z depth size.
          1,   // stride.
          2,   // padding.
          16,  // number of filters.
      });
  model.AddActivationLayer(symbolic::Relu);
  model.AddMaxPoolLayer(/* Input size */ {32, 32, 16},
                        /* output size */ {16, 16});
  model.AddConvolutionLayer(
      {
          16,  // width
          16,  // height
          16,  // R,G,B (depth).
      },
      {
          5,   // filter x size.
          5,   // filter y size.
          16,  // filter z depth size.
          1,   // stride.
          2,   // padding.
          20,  // number of filters.
      });
  model.AddActivationLayer(symbolic::Relu);
  model.AddMaxPoolLayer(/* Input size */ {16, 16, 20},
                        /* output size */ {8, 8});
  model.AddConvolutionLayer(
      {
          8,  // width
          8,  // height
          20,  // R,G,B (depth).
      },
      {
          5,   // filter x size.
          5,   // filter y size.
          20,  // filter z depth size.
          1,   // stride.
          2,   // padding.
          20,  // number of filters.
      });
  model.AddActivationLayer(symbolic::Relu);
  model.AddMaxPoolLayer(/* Input size */ {8, 8, 20},
                        /* output size */ {4, 4});
  model.AddFeedForwardLayer(10);
  model.AddSoftmaxLayer(10);
  nnet::Nnet test_net(model);

  // Need to re-code, this is circle test just for reference.
  //std::vector<std::tuple<Sample, bool>> examples;

  //// Generate training samples.
  //for (size_t i = 0; i < 3000; ++i) {
  //  double x = (2.5 * static_cast<double>(std::rand()) / RAND_MAX) - 1.25;
  //  double y = (2.5 * static_cast<double>(std::rand()) / RAND_MAX) - 1.25;

  //  double in_unit_circle = ((x * x + y * y) <= 1.0) ? 1 : 0;

  //  examples.push_back(std::make_tuple(Sample({{x}, {y}}), in_unit_circle));
  //}

  //nnet::Nnet::LearningParameters params{.learning_rate = 1};

  //for (const std::tuple<Sample, double>& example : examples) {
  //  test_net.TrainCl(std::get<0>(example),
  //                   Matrix<nnet::Number>(
  //                       {{static_cast<nnet::Number>(std::get<1>(example))}}),
  //                   params);
  //}

  //std::cout << std::endl;

  //for (size_t i = 0; i < 1000; ++i) {
  //  double pointx = (2.5 * static_cast<double>(std::rand()) / RAND_MAX) - 1.25;
  //  double pointy = (2.5 * static_cast<double>(std::rand()) / RAND_MAX) - 1.25;
  //  double output =
  //      test_net.EvaluateCl(Matrix<nnet::Number>{{pointx}, {pointy}}).at(0, 0);
  //  std::cout << "((" << pointx << "," << pointy << ")," << output << ")"
  //            << std::endl;
  //}

  //std::cout << std::endl;
  //return 0;
}

#include "math/geometry/matrix.h"
#include "math/nnet/nnet.h"
#include "math/symbolic/expression.h"

#include <cassert>
#include <fstream>
#include <iostream>
#include <string>

constexpr size_t kSampleSize = 2;
using Sample = Matrix<kSampleSize, 1, nnet::Number>;

// Trains a neural network to learn if given point is in unit circle.
int main() {
  constexpr int kNumHiddenLayers = 1;
  constexpr int kLayerSize = 2;
  constexpr int kOutputSize = 1;
  constexpr int kInputSize = kSampleSize;

  using Nnet = nnet::Nnet<kNumHiddenLayers, kLayerSize, kOutputSize, kInputSize>;
  Nnet test_net;

  std::vector<std::tuple<Sample, bool>> examples;

  // Generate training samples.
  for (size_t i = 0; i < 2000; ++i) {
    double x = (4 * static_cast<double>(std::rand()) / RAND_MAX) - 2;
    double y = (4 * static_cast<double>(std::rand()) / RAND_MAX) - 2;

    double in_unit_circle = (x * x + y * y) <= 1.0 ? 1 : 0;

    examples.push_back(std::make_tuple(Sample({{x}, {y}}), in_unit_circle));
  }

  Nnet::LearningParameters params{
      .learning_rate = 0.9
  };

  std::cout << "Training" << std::endl;

  for (const std::tuple<Sample, double>& example : examples) {
    std::cout << "." << std::flush;
    test_net.Train(
        std::get<0>(example),
        Nnet::OutputVector({{static_cast<double>(std::get<1>(example))}}),
        params);
  }

  std::cout << std::endl;

  for (size_t i = 0; i < 1000; ++i) {
    double pointx = (4 * static_cast<double>(std::rand()) / RAND_MAX) - 2;
    double pointy = (4 * static_cast<double>(std::rand()) / RAND_MAX) - 2;
    double output =
        test_net.Evaluate(Nnet::InputVector{{pointx}, {pointy}}).at(0, 0);
    std::cout << "((" << pointx << "," << pointy << ")," << output << ")"
              << std::endl;
  }

  std::cout << test_net.WeightsToString();
  std::cout << std::endl;
  return 0;
}

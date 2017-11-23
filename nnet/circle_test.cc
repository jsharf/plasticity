#include "math/geometry/matrix.h"
#include "math/nnet/nnet.h"
#include "math/symbolic/expression.h"

#include <cassert>
#include <fstream>
#include <iostream>
#include <string>

constexpr size_t kSampleSize = 2;
using Sample = Matrix<kSampleSize, 1, Number>;

// Trains a neural network to learn if given point is in unit circle.
int main() {
  constexpr int kNumHiddenLayers = 0;
  constexpr int kLayerSize = 2;
  constexpr int kOutputSize = 1;
  constexpr int kInputSize = kSampleSize;

  using Nnet = Nnet<kNumHiddenLayers, kLayerSize, kOutputSize, kInputSize>;
  Nnet test_net;
  std::cout << "Expr: " << std::endl << test_net.to_string() << std::endl;

  std::vector<std::tuple<Sample, bool>> examples;

  // Generate training samples.
  for (size_t i = 0; i < 5000; ++i) {
    double x = static_cast<double>(std::rand()) / RAND_MAX;
    double y = static_cast<double>(std::rand()) / RAND_MAX;

    bool in_unit_circle = (x * x + y * y) <= 1.0;

    examples.push_back(std::make_tuple(Sample({{x}, {y}}), in_unit_circle));
  }

  Nnet::LearningParameters params{
      .learning_rate = 0.01,
  };

  std::cout << "Training" << std::endl;

  for (const auto& example : flw) {
    std::cout << "." << std::flush;
    test_net.Train(std::get<0>(example),
                   Nnet::OutputVector({{std::get<1>(example)}}), params);
  }

  std::cout << test_net.WeightsToString();
  std::cout << std::endl;
}

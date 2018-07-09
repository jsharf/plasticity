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
  model.AddFeedForwardLayer(kLayerSize);
  model.AddFeedForwardLayer(kOutputSize);
  nnet::Nnet test_net(model);

  std::vector<std::tuple<Sample, bool>> examples;

  // Generate training samples.
  for (size_t i = 0; i < 3000; ++i) {
    double x = (2.5 * static_cast<double>(std::rand()) / RAND_MAX) - 1.25;
    double y = (2.5 * static_cast<double>(std::rand()) / RAND_MAX) - 1.25;

    double in_unit_circle = ((x * x + y * y) <= 1.0) ? 1 : 0;

    examples.push_back(std::make_tuple(Sample({{x}, {y}}), in_unit_circle));
  }

  nnet::Nnet::LearningParameters params{.learning_rate = 1};

  for (const std::tuple<Sample, double>& example : examples) {
    test_net.TrainCl(std::get<0>(example),
                     Matrix<nnet::Number>(
                         {{static_cast<nnet::Number>(std::get<1>(example))}}),
                     params);
  }

  std::cout << std::endl;

  for (size_t i = 0; i < 1000; ++i) {
    double pointx = (2.5 * static_cast<double>(std::rand()) / RAND_MAX) - 1.25;
    double pointy = (2.5 * static_cast<double>(std::rand()) / RAND_MAX) - 1.25;
    double output_cpu =
        test_net.Evaluate(Matrix<nnet::Number>{{pointx}, {pointy}}).at(0, 0);
    double output_cl =
        test_net.EvaluateCl(Matrix<nnet::Number>{{pointx}, {pointy}}).at(0, 0);
    if (abs(output_cpu - output_cl) > 0.000001) {
      std::cerr << "Error, CPU/GPU output mismatch." << std::endl;
      std::exit(1);
    }
    std::cout << "((" << pointx << "," << pointy << ")," << output_cpu << ")"
              << std::endl;
  }

  std::cout << std::endl;
  return 0;
}

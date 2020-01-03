#include "plasticity/geometry/dynamic_matrix.h"
#include "plasticity/nnet/layer_dimensions.h"
#include "plasticity/nnet/nnet.h"
#include "plasticity/symbolic/expression.h"

#include <cassert>
#include <fstream>
#include <iostream>
#include <string>

constexpr size_t kSampleSize = 2;

// Trains a neural network to learn if given point is in unit circle.
int main() {
  constexpr int kLayerSize = 10;
  constexpr int kOutputSize = 1;
  constexpr int kInputSize = kSampleSize;

  nnet::Architecture model(kInputSize);
  model.AddDenseLayer(kLayerSize);
  model.AddDenseLayer(kOutputSize);
  nnet::Nnet test_net(model, nnet::Nnet::Xavier, nnet::MeanSquared);

  std::vector<std::tuple<std::vector<double>, double>> examples;

  // Generate training samples.
  for (size_t i = 0; i < 10000; ++i) {
    double x = (2.5 * static_cast<double>(std::rand()) / RAND_MAX) - 1.25;
    double y = (2.5 * static_cast<double>(std::rand()) / RAND_MAX) - 1.25;

    double in_unit_circle = ((x * x + y * y) <= 1.0) ? 1 : 0;

    examples.push_back(
        std::make_tuple(std::vector<double>{x, y}, in_unit_circle));
  }

  nnet::Nnet::LearningParameters params{.learning_rate = 0.3};
  test_net.SetLearningParameters(params);

  for (const std::tuple<std::vector<double>, double>& example : examples) {
    auto input = test_net.MakeBuffer(std::get<0>(example));
    std::cout << "AA" << std::endl;
    auto result_buffer = test_net.Evaluate(input);
    std::cout << "A" << std::endl;
    result_buffer->MoveToCpu();
    auto result = (*result_buffer)[0];
    std::cout << "curr output for train example: " << result << std::endl;
    std::cout << "Training w curr output: " << std::get<1>(example)
              << std::endl;
    auto expected_output = test_net.MakeBuffer({std::get<1>(example)});
    test_net.Train(input, expected_output);
  }
  std::cout << "done training!" << std::endl;

  std::cout << std::endl;

  for (size_t i = 0; i < 1000; ++i) {
    double pointx = (2.5 * static_cast<double>(std::rand()) / RAND_MAX) - 1.25;
    double pointy = (2.5 * static_cast<double>(std::rand()) / RAND_MAX) - 1.25;
    auto input_buffer = test_net.MakeBuffer({pointx, pointy});
    auto output_buffer = test_net.Evaluate(input_buffer);
    output_buffer->MoveToCpu();
    double output = (*output_buffer)[0];
    std::cerr << output << std::endl;
    std::cout << "((" << pointx << "," << pointy << ")," << output << ")"
              << std::endl;
  }

  std::cout << std::endl;
  return 0;
}

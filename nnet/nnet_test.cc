#define CATCH_CONFIG_MAIN
#include "math/third_party/catch.h"

#include "math/nnet/nnet.h"

#include <iostream>
#include <string>
#include <memory>

namespace nnet {

using Input = Matrix<Number>;

Input MakeInput(Number a, Number b, Number c) {
  return Input({{a}, {b}, {c}});
}

// This test case generated from the example at:
// https://medium.com/@14prakash/back-propagation-is-very-simple-who-made-it-complicated-97b794c97e5c
TEST_CASE("Simple neural network output is validated", "[nnet]") {

  // This test has a very lenient epsilon compared to regular floating point
  // comparisons because the author of the article (linked above) that this test
  // is based off of used very lenient rounding when calculating the expected
  // outputs.
  constexpr double EPSILON = 0.001;

  constexpr size_t kInputSize = 3;
  constexpr size_t kLayerSize = 3;
  constexpr size_t kOutputSize = 3;

  Architecture model(kInputSize);
  model.AddFeedForwardLayer(kLayerSize, symbolic::Relu);
  model.AddFeedForwardLayer(kLayerSize, symbolic::Sigmoid);
  // (No activation function, next layer is softmax).
  model.AddFeedForwardLayer(kLayerSize, symbolic::Identity);
  model.AddSoftmaxLayer(kOutputSize);

  // All of the layers have identical dimensions (input = output = 3) so we can
  // use the same symbol generator for each.
  FFSymbolGenerator s(Dimensions{3, 3});

  // TODO(sharf): my weight convention (node, edge) is opposite from what the
  // rest of the ML world uses (edge, node).... I should fix this so that this
  // matches the article this is based off better.
  //
  // TODO(sharf): Better way to access and modify layer weights. Jeez this is
  // clunky.

  // Layer 1.
  //
  // Node 1 edges.
  model.layers[1].env()[s.W(0, 0)] = 0.1;
  model.layers[1].env()[s.W(0, 1)] = 0.3;
  model.layers[1].env()[s.W(0, 2)] = 0.4;
  model.layers[1].env()[s.W(0)] = 1;  // bias.
  // Node 2 edges.
  model.layers[1].env()[s.W(1, 0)] = 0.2;
  model.layers[1].env()[s.W(1, 1)] = 0.2;
  model.layers[1].env()[s.W(1, 2)] = 0.3;
  model.layers[1].env()[s.W(1)] = 1;  // bias.
  // Node 3 edges.
  model.layers[1].env()[s.W(2, 0)] = 0.3;
  model.layers[1].env()[s.W(2, 1)] = 0.7;
  model.layers[1].env()[s.W(2, 2)] = 0.9;
  model.layers[1].env()[s.W(2)] = 1;  // bias.

  // Layer 2.
  //
  // Node 1 edges.
  model.layers[2].env()[s.W(0, 0)] = 0.2;
  model.layers[2].env()[s.W(0, 1)] = 0.3;
  model.layers[2].env()[s.W(0, 2)] = 0.6;
  model.layers[2].env()[s.W(0)] = 1;  // bias.
  // Node 2 edges.
  model.layers[2].env()[s.W(1, 0)] = 0.3;
  model.layers[2].env()[s.W(1, 1)] = 0.5;
  model.layers[2].env()[s.W(1, 2)] = 0.4;
  model.layers[2].env()[s.W(1)] = 1;  // bias.
  // Node 3 edges.
  model.layers[2].env()[s.W(2, 0)] = 0.5;
  model.layers[2].env()[s.W(2, 1)] = 0.7;
  model.layers[2].env()[s.W(2, 2)] = 0.8;
  model.layers[2].env()[s.W(2)] = 1;  // bias.

  // Layer 3.
  //
  // Note that there's a separate layer (layer 4) for softmax, so layer 4's
  // outputs should be used to compare to the outputs of layer 3 in the example.
  //
  // Node 1 edges.
  model.layers[3].env()[s.W(0, 0)] = 0.1;
  model.layers[3].env()[s.W(0, 1)] = 0.3;
  model.layers[3].env()[s.W(0, 2)] = 0.5;
  model.layers[3].env()[s.W(0)] = 1;  // bias.

  // Node 2 edges.
  model.layers[3].env()[s.W(1, 0)] = 0.4;
  model.layers[3].env()[s.W(1, 1)] = 0.7;
  model.layers[3].env()[s.W(1, 2)] = 0.2;
  model.layers[3].env()[s.W(1)] = 1;  // bias.

  // Node 3 edges.
  model.layers[3].env()[s.W(2, 0)] = 0.8;
  model.layers[3].env()[s.W(2, 1)] = 0.2;
  model.layers[3].env()[s.W(2, 2)] = 0.9;
  model.layers[3].env()[s.W(2)] = 1;  // bias.


  // Use the model to generate a neural network.
  Nnet test_net(model, Nnet::NoWeightInit, Nnet::CrossEntropy);

  SECTION("Verify output (CPU)") {
    std::unique_ptr<std::vector<Matrix<Number>>> layer_outputs =
        std::make_unique<std::vector<Matrix<Number>>>();
    Matrix<Number> output = test_net.Evaluate(MakeInput(0.1, 0.2, 0.7), layer_outputs);
    REQUIRE(output.dimensions().rows == 3);
    REQUIRE(output.dimensions().cols == 1);
    // Input is counted as a layer and the softmax layer is actually two layers.
    // So 5... up from the 3 in the example.
    REQUIRE(layer_outputs->size() == 5);

    REQUIRE(output.at(0, 0) == Approx(0.19858).epsilon(EPSILON));
    REQUIRE(output.at(1, 0) == Approx(0.28559).epsilon(EPSILON));
    REQUIRE(output.at(2, 0) == Approx(0.51583).epsilon(EPSILON));
  }

  SECTION("Verify output (GPU)") {
    std::unique_ptr<std::vector<Matrix<Number>>> layer_outputs =
        std::make_unique<std::vector<Matrix<Number>>>();
    Matrix<Number> output = test_net.EvaluateCl(MakeInput(0.1, 0.2, 0.7), layer_outputs);
    REQUIRE(output.dimensions().rows == 3);
    REQUIRE(output.dimensions().cols == 1);
    // Input is counted as a layer and the softmax layer is actually two layers.
    // So 5... up from the 3 in the example.
    REQUIRE(layer_outputs->size() == 5);

    REQUIRE(output.at(0, 0) == Approx(0.19858).epsilon(EPSILON));
    REQUIRE(output.at(1, 0) == Approx(0.28559).epsilon(EPSILON));
    REQUIRE(output.at(2, 0) == Approx(0.51583).epsilon(EPSILON));
  }
}

}  // namespace nnet

#define CATCH_CONFIG_MAIN
#include "math/third_party/catch.h"

#include "math/nnet/nnet.h"
#include "math/symbolic/symbolic_util.h"

#include <iostream>
#include <memory>
#include <string>

namespace nnet {

using Input = Matrix<Number>;

Input MakeInput(Number a, Number b, Number c) { return Input({{a}, {b}, {c}}); }
Input MakeInput(Number a, Number b) { return Input({{a}, {b}}); }

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
    Matrix<Number> output =
        test_net.Evaluate(MakeInput(0.1, 0.2, 0.7), layer_outputs);
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
    Matrix<Number> output =
        test_net.EvaluateCl(MakeInput(0.1, 0.2, 0.7), layer_outputs);
    REQUIRE(output.dimensions().rows == 3);
    REQUIRE(output.dimensions().cols == 1);
    // Input is counted as a layer and the softmax layer is actually two layers.
    // So 5... up from the 3 in the example.
    REQUIRE(layer_outputs->size() == 5);

    REQUIRE(output.at(0, 0) == Approx(0.19858).epsilon(EPSILON));
    REQUIRE(output.at(1, 0) == Approx(0.28559).epsilon(EPSILON));
    REQUIRE(output.at(2, 0) == Approx(0.51583).epsilon(EPSILON));
  }

  SECTION("Verify error") {
    // Incorrect actual output values because the example used had a weird
    // calculation for softmax (a/sum(a)) instead of exp(a)/sum(exp(a)).
    Matrix<symbolic::Expression> actual({{0.26980}, {0.32235}, {0.40784}});
    Matrix<symbolic::Expression> expected({{1}, {0}, {0}});

    // Verify error value.
    symbolic::Expression error_value =
        test_net.GenerateCrossEntropyErrorExpression(actual, expected);

    REQUIRE(error_value.Evaluate());
    REQUIRE(error_value.Evaluate()->real() == Approx(0.7410590313576282));
  }
}

// This test case generated from the example at:
// https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
TEST_CASE("Simple neural network output and gradient descent is validated",
          "[nnet]") {
  constexpr size_t kInputSize = 2;
  constexpr size_t kLayerSize = 2;
  constexpr size_t kOutputSize = 2;

  Architecture model(kInputSize);
  model.AddFeedForwardLayer(kLayerSize, symbolic::Sigmoid);
  model.AddFeedForwardLayer(kOutputSize, symbolic::Sigmoid);

  // All of the layers have identical dimensions (input = output = 2) so we
  // can use the same symbol generator for each.
  FFSymbolGenerator s(Dimensions{2, 2});

  // Layer 1.
  //
  // Node 1 edges.
  model.layers[1].env()[s.W(0, 0)] = 0.15;
  model.layers[1].env()[s.W(0, 1)] = 0.2;
  model.layers[1].env()[s.W(0)] = 0.35;  // bias.
  // Node 2 edges.
  model.layers[1].env()[s.W(1, 0)] = 0.25;
  model.layers[1].env()[s.W(1, 1)] = 0.3;
  model.layers[1].env()[s.W(1)] = 0.35;  // bias.

  // Layer 2.
  //
  // Node 1 edges.
  model.layers[2].env()[s.W(0, 0)] = 0.4;
  model.layers[2].env()[s.W(0, 1)] = 0.45;
  model.layers[2].env()[s.W(0)] = 0.60;  // bias.
  // Node 2 edges.
  model.layers[2].env()[s.W(1, 0)] = 0.5;
  model.layers[2].env()[s.W(1, 1)] = 0.55;
  model.layers[2].env()[s.W(1)] = 0.60;  // bias.

  // Use the model to generate a neural network.
  Nnet test_net(model, Nnet::NoWeightInit, Nnet::MeanSquared);

  SECTION("Verify Output of neural network", "[nnet]") {
    std::unique_ptr<std::vector<Matrix<Number>>> layer_outputs =
        std::make_unique<std::vector<Matrix<Number>>>();
    Matrix<Number> output =
        test_net.Evaluate(MakeInput(0.05, 0.10), layer_outputs);
    REQUIRE(output.dimensions().rows == 2);
    REQUIRE(output.dimensions().cols == 1);
    // Input is counted as a layer. So 3... up from the 2 in the example.
    REQUIRE(layer_outputs->size() == 3);

    REQUIRE(output.at(0, 0) == Approx(0.75136507));
    REQUIRE(output.at(1, 0) == Approx(0.772928465));
  }

  Matrix<symbolic::Expression> actual({{0.75136507}, {0.772928465}});
  Matrix<symbolic::Expression> expected({{0.01}, {0.99}});

  SECTION("Verify error propagation for neural network", "[nnet]") {
    symbolic::Expression error =
        test_net.GenerateErrorExpression(actual, expected);
    REQUIRE(error.Evaluate());
    REQUIRE(error.Evaluate()->real() == Approx(0.298371109));
  }

  SECTION("Verify back propagation of neural network", "[nnet]") {
    Matrix<Number> expected({{0.01}, {0.99}});

    SECTION("Verify CPU backprop", "[nnet]") {
      test_net.Train(MakeInput(0.05, 0.10), expected,
                     Nnet::LearningParameters{0.5});
    }

    SECTION("Verify GPU backprop", "[nnet]") {
      test_net.TrainCl(MakeInput(0.05, 0.10), expected,
                     Nnet::LearningParameters{0.5});
    }

    Architecture model = test_net.model();

    // Layer 1.
    //
    // Node 1 edges.
    REQUIRE(model.layers[1].env()[s.W(0, 0)].real() == Approx(0.149780716));
    REQUIRE(model.layers[1].env()[s.W(0, 1)].real() == Approx(0.19956143));
    // Node 2 edges.
    REQUIRE(model.layers[1].env()[s.W(1, 0)].real() == Approx(0.24975114));
    REQUIRE(model.layers[1].env()[s.W(1, 1)].real() == Approx(0.29950229));

    // Layer 2.
    //
    // Node 1 edges.
    REQUIRE(model.layers[2].env()[s.W(0, 0)].real() == Approx(0.35891648));
    REQUIRE(model.layers[2].env()[s.W(0, 1)].real() == Approx(0.408666186));
    // Node 2 edges.
    REQUIRE(model.layers[2].env()[s.W(1, 0)].real() == Approx(0.511301270));
    REQUIRE(model.layers[2].env()[s.W(1, 1)].real() == Approx(0.561370121));
  }
}

}  // namespace nnet

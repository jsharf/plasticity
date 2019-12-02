#define CATCH_CONFIG_MAIN
#include "math/third_party/catch.h"

#include "math/nnet/nnet.h"
#include "math/stats/normal.h"
#include "math/symbolic/symbolic_util.h"

#include <iostream>
#include <memory>
#include <set>
#include <string>

namespace nnet {

// This test case generated from only the first layer in the example at:
// https://medium.com/@14prakash/back-propagation-is-very-simple-who-made-it-complicated-97b794c97e5c
TEST_CASE("One-layer RELU network output is validated", "[nnet]") {
  // This test has a very lenient epsilon compared to regular floating point
  // comparisons because the author of the article (linked above) that this test
  // is based off of used very lenient rounding when calculating the expected
  // outputs.
  constexpr double EPSILON = 0.001;

  constexpr size_t kInputSize = 3;
  constexpr size_t kLayerSize = 3;

  Architecture model(kInputSize);
  model.AddDenseLayer(kLayerSize, symbolic::Relu);

  // Use the model to generate a neural network.
  Nnet test_net(model, Nnet::NoWeightInit, CrossEntropy);

  DenseSymbolGenerator s(Dimensions{3, 3});

  // Layer 1.
  //
  // The weights used in layer 1 in this example are actually misprinted at one
  // point in the article. They are printed correctly at the beginning & end
  // however, and in the area with the misprint, the results match what you
  // would expect using the correct weights, so it looks like it's just a
  // harmless (yet confusing) mistake.
  //
  // Node 1 edges.
  test_net.GetWeight(1, s.WeightNumber(0, 0)) = 0.1;
  test_net.GetWeight(1, s.WeightNumber(0, 1)) = 0.3;
  test_net.GetWeight(1, s.WeightNumber(0, 2)) = 0.4;
  test_net.GetWeight(1, s.WeightNumber(0)) = 1;  // bias.
  // Node 2 edges.
  test_net.GetWeight(1, s.WeightNumber(1, 0)) = 0.2;
  test_net.GetWeight(1, s.WeightNumber(1, 1)) = 0.2;
  test_net.GetWeight(1, s.WeightNumber(1, 2)) = 0.3;
  test_net.GetWeight(1, s.WeightNumber(1)) = 1;  // bias.
  // Node 3 edges.
  test_net.GetWeight(1, s.WeightNumber(2, 0)) = 0.3;
  test_net.GetWeight(1, s.WeightNumber(2, 1)) = 0.7;
  test_net.GetWeight(1, s.WeightNumber(2, 2)) = 0.9;
  test_net.GetWeight(1, s.WeightNumber(2)) = 1;  // bias.

  SECTION("Verify output") {
    auto input = test_net.MakeBuffer({0.1, 0.2, 0.7});
    auto output = test_net.Evaluate(input);
    output->MoveToCpu();

    REQUIRE(output->size() == 3);

    REQUIRE(output->at(0) == Approx(1.35).epsilon(EPSILON));
    REQUIRE(output->at(1) == Approx(1.27).epsilon(EPSILON));
    REQUIRE(output->at(2) == Approx(1.8).epsilon(EPSILON));
  }
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
  model.AddDenseLayer(kLayerSize, symbolic::Relu);
  model.AddDenseLayer(kLayerSize, symbolic::Sigmoid);
  // (No activation function, next layer is softmax).
  model.AddDenseLayer(kLayerSize, symbolic::Identity);
  model.AddSoftmaxLayer(kOutputSize);

  // All of the layers have identical dimensions (input = output = 3) so we can
  // use the same symbol generator for each.
  DenseSymbolGenerator s(Dimensions{3, 3});

  // TODO(sharf): my weight convention (node, edge) is opposite from what the
  // rest of the ML world uses (edge, node).... I should fix this so that this
  // matches the article this is based off better.
  //
  // TODO(sharf): Better way to access and modify layer weights. Jeez this is
  // clunky.

  // Layer 1.
  //
  // The weights used in layer 1 in this example are actually misprinted at one
  // point in the article. They are printed correctly at the beginning & end
  // however, and in the area with the misprint, the results match what you
  // would expect using the correct weights, so it looks like it's just a
  // harmless (yet confusing) mistake.
  //
  // Node 1 edges.
  model.layers[1].W(s.WeightNumber(0, 0)) = 0.1;
  model.layers[1].W(s.WeightNumber(0, 1)) = 0.3;
  model.layers[1].W(s.WeightNumber(0, 2)) = 0.4;
  model.layers[1].W(s.WeightNumber(0)) = 1;  // bias.
  // Node 2 edges.
  model.layers[1].W(s.WeightNumber(1, 0)) = 0.2;
  model.layers[1].W(s.WeightNumber(1, 1)) = 0.2;
  model.layers[1].W(s.WeightNumber(1, 2)) = 0.3;
  model.layers[1].W(s.WeightNumber(1)) = 1;  // bias.
  // Node 3 edges.
  model.layers[1].W(s.WeightNumber(2, 0)) = 0.3;
  model.layers[1].W(s.WeightNumber(2, 1)) = 0.7;
  model.layers[1].W(s.WeightNumber(2, 2)) = 0.9;
  model.layers[1].W(s.WeightNumber(2)) = 1;  // bias.

  // Layer 2.
  //
  // Node 1 edges.
  model.layers[3].W(s.WeightNumber(0, 0)) = 0.2;
  model.layers[3].W(s.WeightNumber(0, 1)) = 0.3;
  model.layers[3].W(s.WeightNumber(0, 2)) = 0.6;
  model.layers[3].W(s.WeightNumber(0)) = 1;  // bias.
  // Node 2 edges.
  model.layers[3].W(s.WeightNumber(1, 0)) = 0.3;
  model.layers[3].W(s.WeightNumber(1, 1)) = 0.5;
  model.layers[3].W(s.WeightNumber(1, 2)) = 0.4;
  model.layers[3].W(s.WeightNumber(1)) = 1;  // bias.
  // Node 3 edges.
  model.layers[3].W(s.WeightNumber(2, 0)) = 0.5;
  model.layers[3].W(s.WeightNumber(2, 1)) = 0.7;
  model.layers[3].W(s.WeightNumber(2, 2)) = 0.8;
  model.layers[3].W(s.WeightNumber(2)) = 1;  // bias.

  // Layer 3.
  //
  // Note that there's a separate layer (layer 4) for softmax, so layer 4's
  // outputs should be used to compare to the outputs of layer 3 in the example.
  //
  // Node 1 edges.
  model.layers[5].W(s.WeightNumber(0, 0)) = 0.1;
  model.layers[5].W(s.WeightNumber(0, 1)) = 0.3;
  model.layers[5].W(s.WeightNumber(0, 2)) = 0.5;
  model.layers[5].W(s.WeightNumber(0)) = 1;  // bias.

  // Node 2 edges.
  model.layers[5].W(s.WeightNumber(1, 0)) = 0.4;
  model.layers[5].W(s.WeightNumber(1, 1)) = 0.7;
  model.layers[5].W(s.WeightNumber(1, 2)) = 0.2;
  model.layers[5].W(s.WeightNumber(1)) = 1;  // bias.

  // Node 3 edges.
  model.layers[5].W(s.WeightNumber(2, 0)) = 0.8;
  model.layers[5].W(s.WeightNumber(2, 1)) = 0.2;
  model.layers[5].W(s.WeightNumber(2, 2)) = 0.9;
  model.layers[5].W(s.WeightNumber(2)) = 1;  // bias.

  // Use the model to generate a neural network.
  Nnet test_net(model, Nnet::NoWeightInit, CrossEntropy);

  SECTION("Verify output") {
    auto output = test_net.Evaluate(test_net.MakeBuffer({0.1, 0.2, 0.7}));
    output->MoveToCpu();

    REQUIRE(output->size() == 3);

    REQUIRE(output->at(0) == Approx(0.19858).epsilon(EPSILON));
    REQUIRE(output->at(1) == Approx(0.28559).epsilon(EPSILON));
    REQUIRE(output->at(2) == Approx(0.51583).epsilon(EPSILON));
  }

  SECTION("Verify error") {
    // Incorrect actual output values because the example used had a weird
    // calculation for softmax (a/sum(a)) instead of exp(a)/sum(exp(a)).
    auto actual = test_net.MakeBuffer({0.26980, 0.32235, 0.40784});
    auto expected = test_net.MakeBuffer({1.0, 0.0, 0.0});

    REQUIRE(test_net.Error(actual, expected) == Approx(1.310074));
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
  model.AddDenseLayer(kLayerSize, symbolic::Sigmoid);
  model.AddDenseLayer(kOutputSize, symbolic::Sigmoid);

  // All of the layers have identical dimensions (input = output = 2) so we
  // can use the same symbol generator for each.
  DenseSymbolGenerator s(Dimensions{2, 2});

  // Layer 1.
  //
  // Node 1 edges.
  model.layers[1].W(s.WeightNumber(0, 0)) = 0.15;
  model.layers[1].W(s.WeightNumber(0, 1)) = 0.2;
  model.layers[1].W(s.WeightNumber(0)) = 0.35;  // bias.
  // Node 2 edges.
  model.layers[1].W(s.WeightNumber(1, 0)) = 0.25;
  model.layers[1].W(s.WeightNumber(1, 1)) = 0.3;
  model.layers[1].W(s.WeightNumber(1)) = 0.35;  // bias.

  // Layer 2.
  //
  // Node 1 edges.
  model.layers[3].W(s.WeightNumber(0, 0)) = 0.4;
  model.layers[3].W(s.WeightNumber(0, 1)) = 0.45;
  model.layers[3].W(s.WeightNumber(0)) = 0.60;  // bias.
  // Node 2 edges.
  model.layers[3].W(s.WeightNumber(1, 0)) = 0.5;
  model.layers[3].W(s.WeightNumber(1, 1)) = 0.55;
  model.layers[3].W(s.WeightNumber(1)) = 0.60;  // bias.

  // Use the model to generate a neural network.
  Nnet test_net(model, Nnet::NoWeightInit, MeanSquared);

  SECTION("Verify Output of neural network", "[nnet]") {
    auto output = test_net.Evaluate(test_net.MakeBuffer({0.05, 0.10}));
    REQUIRE(output->size() == 2);
    output->MoveToCpu();

    REQUIRE(output->at(0) == Approx(0.75136507));
    REQUIRE(output->at(1) == Approx(0.772928465));
  }

  SECTION("Verify error propagation for neural network", "[nnet]") {
    auto actual = test_net.MakeBuffer({0.75136507, 0.772928465});
    auto expected = test_net.MakeBuffer({0.01, 0.99});

    REQUIRE(test_net.Error(actual, expected) == Approx(0.298371109));
  }

  SECTION("Verify back propagation of neural network", "[nnet]") {
    std::cerr << "HIII" << std::endl;
    auto expected = test_net.MakeBuffer({0.01, 0.99});
    std::cerr << "BEEE" << std::endl;
    auto input = test_net.MakeBuffer({0.05, 0.10});
    std::cerr << "CE" << std::endl;
    test_net.SetLearningParameters(Nnet::LearningParameters{0.5});
    std::cerr << "D" << std::endl;
    test_net.Train(input, expected);
    std::cout << "Going to sleep..." << std::endl;
    sleep(10);
    std::cout << "wokeup" << std::endl;
    std::cerr << "E" << std::endl;

    Architecture model = test_net.model();
    std::cerr << "F" << std::endl;

    // Layer 1.
    //
    // Node 1 edges.
    CHECK(model.layers[1].W(s.WeightNumber(0, 0)) == Approx(0.149780716));
    CHECK(model.layers[1].W(s.WeightNumber(0, 1)) == Approx(0.19956143));
    // Node 2 edges.
    CHECK(model.layers[1].W(s.WeightNumber(1, 0)) == Approx(0.24975114));
    CHECK(model.layers[1].W(s.WeightNumber(1, 1)) == Approx(0.29950229));

    // Layer 2.
    //
    // Node 1 edges.
    CHECK(model.layers[3].W(s.WeightNumber(0, 0)) == Approx(0.35891648));
    CHECK(model.layers[3].W(s.WeightNumber(0, 1)) == Approx(0.408666186));
    // Node 2 edges.
    CHECK(model.layers[3].W(s.WeightNumber(1, 0)) == Approx(0.511301270));
    CHECK(model.layers[3].W(s.WeightNumber(1, 1)) == Approx(0.561370121));
  }
}

TEST_CASE("Just testing a single max_pool layer", "[maxpool]") {
  constexpr size_t kInputSize = 48;

  Architecture model(kInputSize);
  model.AddMaxPoolLayer(
      // Input
      {
          4, 4, 3  // width, height, depth
      },
      // Output (depth not specified since it's assumed to be the same as
      // input).
      {
          2, 2  // width, height
      });

  // Use the model to generate a neural network.
  Nnet test_net(model, Nnet::NoWeightInit, MeanSquared);

  // input is a 3D 4x4x3 image.
  auto example = test_net.MakeBuffer({
      // Layer 1
      1, 0, 0, 3, 32, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 5,
      // Layer 2
      1, 0, 0, 3, 7, 8, 10, 2, 0, 0, 5, 0, 1, 0, 0, 5,
      // Layer 3
      1, -8, 0, 3, 32, 24, 0, 2, 0, 100, 0, 0, 0, 0, 0, 5,
  });

  auto expected = test_net.MakeBuffer({
      // Layer 1
      32, 3, 0, 5,
      // Layer 2
      8, 10, 1, 5,
      // Layer 3
      32, 3, 100, 5,
  });

  auto actual = test_net.Evaluate(example);

  actual->MoveToCpu();
  expected->MoveToCpu();

  SECTION("forward pass", "[maxpool]") {
    for (size_t i = 0; i < expected->size(); ++i) {
      CAPTURE(expected->at(i));
      CAPTURE(actual->at(i));
      REQUIRE(expected->at(i) == Approx(actual->at(i)));
    }
  }

  SECTION("gradient backprop", "[maxpool]") {
    nnet::Nnet::LearningParameters params{.learning_rate = 1};
    auto expected_altered = test_net.MakeBuffer({
        // Layer 1
        33, 3, 100, 4,
        // Layer 2
        8, 10, 1, 5,
        // Layer 3
        31, 4, 90, 6,
    });
    std::unique_ptr<compute::ClBuffer> gradients = test_net.MakeBuffer(0);
    test_net.SetLearningParameters(params);
    test_net.Train(example, expected_altered, gradients);

    gradients->MoveToCpu();
    example->MoveToCpu();

    REQUIRE(gradients->size() == 48);
    REQUIRE(example->size() == 48);

    // Layer 1
    CHECK(gradients->at(0) == Approx(0.0));
    CHECK(gradients->at(1) == Approx(0.0));
    CHECK(gradients->at(2) == Approx(0.0));
    CHECK(gradients->at(3) == Approx(0.0));

    CHECK(gradients->at(4) < (0.0));
    CHECK(gradients->at(5) == Approx(0.0));
    CHECK(gradients->at(6) == Approx(0.0));
    CHECK(gradients->at(7) == Approx(0.0));

    CHECK(gradients->at(8) < (0.0));
    CHECK(gradients->at(9) < (0.0));
    CHECK(gradients->at(10) == Approx(0.0));
    CHECK(gradients->at(11) == Approx(0.0));

    CHECK(gradients->at(12) < (0.0));
    CHECK(gradients->at(13) < (0.0));
    CHECK(gradients->at(14) == Approx(0.0));
    CHECK(gradients->at(15) > (0.0));

    // Layer 2
    CHECK(gradients->at(16) == Approx(0.0));
    CHECK(gradients->at(17) == Approx(0.0));
    CHECK(gradients->at(18) == Approx(0.0));
    CHECK(gradients->at(19) == Approx(0.0));

    CHECK(gradients->at(20) == Approx(0.0));
    CHECK(gradients->at(21) == Approx(0.0));
    CHECK(gradients->at(22) == Approx(0.0));
    CHECK(gradients->at(23) == Approx(0.0));

    CHECK(gradients->at(24) == Approx(0.0));
    CHECK(gradients->at(25) == Approx(0.0));
    CHECK(gradients->at(26) == Approx(0.0));
    CHECK(gradients->at(27) == Approx(0.0));

    CHECK(gradients->at(28) == Approx(0.0));
    CHECK(gradients->at(29) == Approx(0.0));
    CHECK(gradients->at(30) == Approx(0.0));
    CHECK(gradients->at(31) == Approx(0.0));

    // Layer 3
    CHECK(gradients->at(32) == Approx(0.0));
    CHECK(gradients->at(33) == Approx(0.0));
    CHECK(gradients->at(34) == Approx(0.0));
    CHECK(gradients->at(35) < (0.0));

    CHECK(gradients->at(36) > (0.0));
    CHECK(gradients->at(37) == Approx(0.0));
    CHECK(gradients->at(38) == Approx(0.0));
    CHECK(gradients->at(39) == Approx(0.0));

    CHECK(gradients->at(40) == Approx(0.0));
    CHECK(gradients->at(41) > (0.0));
    CHECK(gradients->at(42) == Approx(0.0));
    CHECK(gradients->at(43) == Approx(0.0));

    CHECK(gradients->at(44) == Approx(0.0));
    CHECK(gradients->at(45) == Approx(0.0));
    CHECK(gradients->at(46) == Approx(0.0));
    CHECK(gradients->at(47) < (0.0));
  }
}

TEST_CASE("Symbol Generator", "[symgen]") {
  SECTION("Convolution symbol generator", "[symgen]") {
    VolumeDimensions layer_dimensions = {
        5, 5, 3  // width, height, depth
    };

    FilterParams params = {
        3, 3, 3,  // Width, Height, Depth
        2,        // Stride
        1,        // Padding
        2         // Number of filters.
    };

    ConvSymbolGenerator s(layer_dimensions, params);

    // Layer 1.
    REQUIRE(s.WeightNumber(0, 0, 0, 0) == 0);
    REQUIRE(s.WeightNumber(0, 0, 1, 0) == 1);
    REQUIRE(s.WeightNumber(0, 0, 2, 0) == 2);

    REQUIRE(s.WeightNumber(0, 1, 0, 0) == 3);
    REQUIRE(s.WeightNumber(0, 1, 1, 0) == 4);
    REQUIRE(s.WeightNumber(0, 1, 2, 0) == 5);

    REQUIRE(s.WeightNumber(0, 2, 0, 0) == 6);
    REQUIRE(s.WeightNumber(0, 2, 1, 0) == 7);
    REQUIRE(s.WeightNumber(0, 2, 2, 0) == 8);

    REQUIRE(s.WeightNumber(0, 0, 0, 1) == 9);
    REQUIRE(s.WeightNumber(0, 0, 1, 1) == 10);
    REQUIRE(s.WeightNumber(0, 0, 2, 1) == 11);

    REQUIRE(s.WeightNumber(0, 1, 0, 1) == 12);
    REQUIRE(s.WeightNumber(0, 1, 1, 1) == 13);
    REQUIRE(s.WeightNumber(0, 1, 2, 1) == 14);

    REQUIRE(s.WeightNumber(0, 2, 0, 1) == 15);
    REQUIRE(s.WeightNumber(0, 2, 1, 1) == 16);
    REQUIRE(s.WeightNumber(0, 2, 2, 1) == 17);

    REQUIRE(s.WeightNumber(0, 0, 0, 2) == 18);
    REQUIRE(s.WeightNumber(0, 0, 1, 2) == 19);
    REQUIRE(s.WeightNumber(0, 0, 2, 2) == 20);

    REQUIRE(s.WeightNumber(0, 1, 0, 2) == 21);
    REQUIRE(s.WeightNumber(0, 1, 1, 2) == 22);
    REQUIRE(s.WeightNumber(0, 1, 2, 2) == 23);

    REQUIRE(s.WeightNumber(0, 2, 0, 2) == 24);
    REQUIRE(s.WeightNumber(0, 2, 1, 2) == 25);
    REQUIRE(s.WeightNumber(0, 2, 2, 2) == 26);

    // Bias
    REQUIRE(s.WeightNumber(0) == 27);

    // Layer 2.
    REQUIRE(s.WeightNumber(1, 0, 0, 0) == 28);
    REQUIRE(s.WeightNumber(1, 0, 1, 0) == 29);
    REQUIRE(s.WeightNumber(1, 0, 2, 0) == 30);

    REQUIRE(s.WeightNumber(1, 1, 0, 0) == 31);
    REQUIRE(s.WeightNumber(1, 1, 1, 0) == 32);
    REQUIRE(s.WeightNumber(1, 1, 2, 0) == 33);

    REQUIRE(s.WeightNumber(1, 2, 0, 0) == 34);
    REQUIRE(s.WeightNumber(1, 2, 1, 0) == 35);
    REQUIRE(s.WeightNumber(1, 2, 2, 0) == 36);

    REQUIRE(s.WeightNumber(1, 0, 0, 1) == 37);
    REQUIRE(s.WeightNumber(1, 0, 1, 1) == 38);
    REQUIRE(s.WeightNumber(1, 0, 2, 1) == 39);

    REQUIRE(s.WeightNumber(1, 1, 0, 1) == 40);
    REQUIRE(s.WeightNumber(1, 1, 1, 1) == 41);
    REQUIRE(s.WeightNumber(1, 1, 2, 1) == 42);

    REQUIRE(s.WeightNumber(1, 2, 0, 1) == 43);
    REQUIRE(s.WeightNumber(1, 2, 1, 1) == 44);
    REQUIRE(s.WeightNumber(1, 2, 2, 1) == 45);

    REQUIRE(s.WeightNumber(1, 0, 0, 2) == 46);
    REQUIRE(s.WeightNumber(1, 0, 1, 2) == 47);
    REQUIRE(s.WeightNumber(1, 0, 2, 2) == 48);

    REQUIRE(s.WeightNumber(1, 1, 0, 2) == 49);
    REQUIRE(s.WeightNumber(1, 1, 1, 2) == 50);
    REQUIRE(s.WeightNumber(1, 1, 2, 2) == 51);

    REQUIRE(s.WeightNumber(1, 2, 0, 2) == 52);
    REQUIRE(s.WeightNumber(1, 2, 1, 2) == 53);
    REQUIRE(s.WeightNumber(1, 2, 2, 2) == 54);

    // Bias
    REQUIRE(s.WeightNumber(1) == 55);
  }
}

// Taken from the convolution demo here:
// http://cs231n.github.io/convolutional-networks/#conv
TEST_CASE("Convolution layer test", "[convnet]") {
  constexpr size_t kInputSize = 75;

  Architecture model(kInputSize);
  VolumeDimensions layer_dimensions = {
      5, 5, 3  // width, height, depth
  };
  FilterParams params = {
      3, 3, 3,  // Width, Height, Depth
      2,        // Stride
      1,        // Padding
      2         // Number of filters.
  };
  model.AddConvolutionLayer(
      // Inputs
      layer_dimensions,
      // Filter params
      params,
      // No activation function.
      symbolic::Identity);

  ConvSymbolGenerator s(layer_dimensions, params);

  // Filter 1.
  //
  // Depth 0
  model.layers[1].W(s.WeightNumber(0, 0, 0, 0)) = 0;
  model.layers[1].W(s.WeightNumber(0, 0, 1, 0)) = 1;
  model.layers[1].W(s.WeightNumber(0, 0, 2, 0)) = 0;

  model.layers[1].W(s.WeightNumber(0, 1, 0, 0)) = 0;
  model.layers[1].W(s.WeightNumber(0, 1, 1, 0)) = -1;
  model.layers[1].W(s.WeightNumber(0, 1, 2, 0)) = 1;

  model.layers[1].W(s.WeightNumber(0, 2, 0, 0)) = -1;
  model.layers[1].W(s.WeightNumber(0, 2, 1, 0)) = 1;
  model.layers[1].W(s.WeightNumber(0, 2, 2, 0)) = 0;

  // Depth 1
  model.layers[1].W(s.WeightNumber(0, 0, 0, 1)) = 0;
  model.layers[1].W(s.WeightNumber(0, 0, 1, 1)) = -1;
  model.layers[1].W(s.WeightNumber(0, 0, 2, 1)) = 1;

  model.layers[1].W(s.WeightNumber(0, 1, 0, 1)) = 0;
  model.layers[1].W(s.WeightNumber(0, 1, 1, 1)) = -1;
  model.layers[1].W(s.WeightNumber(0, 1, 2, 1)) = -1;

  model.layers[1].W(s.WeightNumber(0, 2, 0, 1)) = 0;
  model.layers[1].W(s.WeightNumber(0, 2, 1, 1)) = 0;
  model.layers[1].W(s.WeightNumber(0, 2, 2, 1)) = -1;

  // Depth 3
  model.layers[1].W(s.WeightNumber(0, 0, 0, 2)) = -1;
  model.layers[1].W(s.WeightNumber(0, 0, 1, 2)) = -1;
  model.layers[1].W(s.WeightNumber(0, 0, 2, 2)) = 1;

  model.layers[1].W(s.WeightNumber(0, 1, 0, 2)) = 1;
  model.layers[1].W(s.WeightNumber(0, 1, 1, 2)) = 1;
  model.layers[1].W(s.WeightNumber(0, 1, 2, 2)) = 0;

  model.layers[1].W(s.WeightNumber(0, 2, 0, 2)) = 0;
  model.layers[1].W(s.WeightNumber(0, 2, 1, 2)) = 1;
  model.layers[1].W(s.WeightNumber(0, 2, 2, 2)) = 1;

  // Bias
  model.layers[1].W(s.WeightNumber(0)) = 1;

  // Filter 2.
  //
  // Depth 0
  model.layers[1].W(s.WeightNumber(1, 0, 0, 0)) = 0;
  model.layers[1].W(s.WeightNumber(1, 0, 1, 0)) = -1;
  model.layers[1].W(s.WeightNumber(1, 0, 2, 0)) = -1;

  model.layers[1].W(s.WeightNumber(1, 1, 0, 0)) = 1;
  model.layers[1].W(s.WeightNumber(1, 1, 1, 0)) = 1;
  model.layers[1].W(s.WeightNumber(1, 1, 2, 0)) = 0;

  model.layers[1].W(s.WeightNumber(1, 2, 0, 0)) = 0;
  model.layers[1].W(s.WeightNumber(1, 2, 1, 0)) = 0;
  model.layers[1].W(s.WeightNumber(1, 2, 2, 0)) = 0;

  // Depth 1
  model.layers[1].W(s.WeightNumber(1, 0, 0, 1)) = 0;
  model.layers[1].W(s.WeightNumber(1, 0, 1, 1)) = -1;
  model.layers[1].W(s.WeightNumber(1, 0, 2, 1)) = -1;

  model.layers[1].W(s.WeightNumber(1, 1, 0, 1)) = 0;
  model.layers[1].W(s.WeightNumber(1, 1, 1, 1)) = 0;
  model.layers[1].W(s.WeightNumber(1, 1, 2, 1)) = 0;

  model.layers[1].W(s.WeightNumber(1, 2, 0, 1)) = 1;
  model.layers[1].W(s.WeightNumber(1, 2, 1, 1)) = 0;
  model.layers[1].W(s.WeightNumber(1, 2, 2, 1)) = -1;

  // Depth 3
  model.layers[1].W(s.WeightNumber(1, 0, 0, 2)) = 1;
  model.layers[1].W(s.WeightNumber(1, 0, 1, 2)) = 1;
  model.layers[1].W(s.WeightNumber(1, 0, 2, 2)) = -1;

  model.layers[1].W(s.WeightNumber(1, 1, 0, 2)) = 1;
  model.layers[1].W(s.WeightNumber(1, 1, 1, 2)) = 1;
  model.layers[1].W(s.WeightNumber(1, 1, 2, 2)) = 0;

  model.layers[1].W(s.WeightNumber(1, 2, 0, 2)) = 1;
  model.layers[1].W(s.WeightNumber(1, 2, 1, 2)) = 0;
  model.layers[1].W(s.WeightNumber(1, 2, 2, 2)) = 0;

  // Bias
  model.layers[1].W(s.WeightNumber(1)) = 0;

  // Use the model to generate a neural network.
  Nnet test_net(model, Nnet::NoWeightInit, MeanSquared);

  // input is a 3D 5x5x3 image.
  auto example = test_net.MakeBuffer({
      // Layer 1
      1, 0, 0, 1, 1, 2, 0, 0, 2, 1, 0, 1, 1, 2, 0, 1, 0, 0, 2, 1, 2, 2, 1, 1, 1,
      // Layer 2
      0, 2, 0, 0, 1, 1, 1, 1, 1, 2, 1, 1, 0, 1, 1, 1, 1, 1, 0, 2, 2, 0, 2, 0, 0,
      // Layer 3
      2, 0, 2, 1, 2, 0, 2, 1, 0, 0, 2, 1, 1, 0, 0, 1, 1, 0, 0, 0, 2, 1, 1, 1, 2,
  });

  auto expected = test_net.MakeBuffer({
      // Layer 1
      3, 4, 1, 8, 0, -2, 2, -1, 2,
      // Layer 2
      2, 4, 6, -5, 5, -1, 1, 3, 2,
  });

  auto actual = test_net.Evaluate(example);

  actual->MoveToCpu();
  SECTION("forward pass", "[convnet]") {
    for (size_t i = 0; i < expected->size(); ++i) {
      CAPTURE(i);
      CAPTURE(expected->at(i));
      CAPTURE(actual->at(i));
      CHECK(expected->at(i) == Approx(actual->at(i)));
    }
  }

  SECTION("training pass", "[convnet]") {
    auto expected_altered = test_net.MakeBuffer({
        // Layer 1
        4, 4, 1, 8, 0, -2, 2, -1, 2,
        // Layer 2
        2, 4, 6, -5, 5, -1, 1, 3, 2,
    });

    nnet::Nnet::LearningParameters params{.learning_rate = 1};
    std::unique_ptr<compute::ClBuffer> gradients = test_net.MakeBuffer(0);
    test_net.SetLearningParameters(params);
    test_net.Train(example, expected_altered, gradients);

    gradients->MoveToCpu();
    example->MoveToCpu();

    REQUIRE(gradients->size() == 75);
    REQUIRE(example->size() == 75);

    // Expected non-zero gradients.
    // Visually , I expected 30 & 51 to also be non-zero, however actually doing
    // the math showed that the gradient works out to zero, either because
    // differing gradients cancel out or because of the weight values. This was
    // verified with heavy debugging, and running the kernel source code as a C
    // program (printing it out and copy-pasting it to a file, then compiling)
    // and adding a bunch of printf statements to confirm. See the independent
    // C++ file "example_kernel.cc" for this source code. You can change the
    // independent index to 51 and the z-variable to 2 to confirm for 51, or 30
    // & 1 to confirm for 30.
    std::set<int> expected_changed_gradients = {
        // Layer 1.
        0, 1, 5,
        // Layer 2.
        25, 26, 31,
        // Layer 3.
        50, 55, 56,
    };

    for (size_t i = 0; i < 75; ++i) {
      if (expected_changed_gradients.count(i) == 1) {
        CAPTURE(i);
        REQUIRE(gradients->at(i) != (0.0));
      } else {
        REQUIRE(gradients->at(i) == Approx(0.0));
      }
    }
  }
}

TEST_CASE("Dense Layer Gradient checking", "[densenet]") {
  constexpr double EPSILON = 0.000001;
  constexpr double COMPARISON_EPSILON = 0.01;

  constexpr size_t kInputSize = 5;
  constexpr size_t kLayerSize = 5;

  Architecture model(kInputSize);
  model.AddDenseLayer(kLayerSize, symbolic::Identity);

  stats::Normal initializer(0, 5);
  std::unique_ptr<compute::ClBuffer> input =
      std::make_unique<compute::ClBuffer>(5);
  for (size_t i = 0; i < 5; ++i) {
    input->at(i) = initializer.sample();
  }

  SECTION("Verify input gradient") {
    // Use the model to generate a neural network.
    Nnet test_net(model, Nnet::Xavier, MeanSquared);
    auto expected = test_net.MakeBuffer({0.2, 0.2, 0.2, 0.2, 0.2});
    test_net.RegisterBuffer(input.get());

    nnet::Nnet::LearningParameters params{.learning_rate = 0};
    std::unique_ptr<compute::ClBuffer> gradients = test_net.MakeBuffer(0);
    test_net.SetLearningParameters(params);
    test_net.Train(input, expected, gradients);

    gradients->MoveToCpu();
    input->MoveToCpu();

    for (size_t i = 0; i < 5; ++i) {
      std::unique_ptr<compute::ClBuffer> input_left =
          std::make_unique<compute::ClBuffer>(*input);
      input_left->at(i) -= EPSILON;
      std::unique_ptr<compute::ClBuffer> input_right =
          std::make_unique<compute::ClBuffer>(*input);
      input_right->at(i) += EPSILON;
      auto output_left = test_net.Evaluate(input_left);
      double error_left = test_net.Error(output_left, expected);
      auto output_right = test_net.Evaluate(input_right);
      double error_right = test_net.Error(output_right, expected);

      double approx_gradient = (error_right - error_left) / (2 * EPSILON);
      double actual_gradient = gradients->at(i);

      CAPTURE(i);
      REQUIRE(actual_gradient == Approx(approx_gradient).epsilon(EPSILON));
    }
  }

  SECTION("Verify weight gradient") {
    DenseSymbolGenerator s(Dimensions{9, 9});
    nnet::Nnet::LearningParameters params{.learning_rate = 1};

    stats::Normal initializer(0, 5);
    std::vector<double> weight_init_values(
        model.layers[1].weight_buffer().size());
    for (size_t i = 0; i < model.layers[1].weight_buffer().size(); ++i) {
      model.layers[1].weight_buffer()[i] = initializer.sample();
      weight_init_values[i] = model.layers[1].weight_buffer()[i];
    }

    for (size_t i = 0; i < model.layers[1].weight_buffer().size(); ++i) {
      // A neural network with the weight tweaked left.
      model.layers[1].weight_buffer().MoveToCpu();
      model.layers[1].weight_buffer()[i] = weight_init_values[i] - EPSILON;
      Nnet test_net_tweak_left(model, Nnet::NoWeightInit, MeanSquared);
      auto expected = test_net_tweak_left.MakeBuffer({0.2, 0.2, 0.2, 0.2, 0.2});
      test_net_tweak_left.RegisterBuffer(input.get());

      auto output_left = test_net_tweak_left.Evaluate(input);
      double error_left = test_net_tweak_left.Error(output_left, expected);

      // A neural network with the weight tweaked right.
      model.layers[1].weight_buffer().MoveToCpu();
      model.layers[1].weight_buffer()[i] = weight_init_values[i] + EPSILON;
      Nnet test_net_tweak_right(model, Nnet::NoWeightInit, MeanSquared);
      test_net_tweak_right.RegisterBuffer(input.get());
      expected = test_net_tweak_right.MakeBuffer({0.2, 0.2, 0.2, 0.2, 0.2});

      auto output_right = test_net_tweak_right.Evaluate(input);
      double error_right = test_net_tweak_right.Error(output_right, expected);

      double approx_gradient = (error_right - error_left) / (2 * EPSILON);

      model.layers[1].weight_buffer().MoveToCpu();
      model.layers[1].weight_buffer()[i] = weight_init_values[i];
      Nnet test_net(model, Nnet::NoWeightInit, MeanSquared);

      test_net.RegisterBuffer(input.get());
      auto result = test_net.Evaluate(input);
      test_net.RegisterBuffer(expected.get());
      test_net.SetLearningParameters(params);
      test_net.Train(input, expected);
      double weight_gradient = weight_init_values[i] - test_net.GetWeight(1, i);

      CAPTURE(i);
      REQUIRE(weight_gradient ==
              Approx(approx_gradient).epsilon(COMPARISON_EPSILON));

      input->MoveToCpu();
      expected->MoveToCpu();
    }
  }
}

TEST_CASE("Convolution Layer Gradient checking",
          "[convolution_gradient_check]") {
  constexpr double EPSILON = 0.000001;
  constexpr double COMPARISON_EPSILON = 0.0001;

  constexpr size_t kInputSize = 9;

  Architecture model(kInputSize);
  model.AddConvolutionLayer({3, 3, 1}, {3, 3, 1, 1, 1, 1}, symbolic::Relu);
  Nnet test_net(model, Nnet::Xavier, CrossEntropy);

  stats::Normal initializer(0, 5);
  auto input = test_net.MakeBuffer(9);
  for (size_t i = 0; i < 9; ++i) {
    input->at(i) = initializer.sample();
  }

  auto expected = test_net.MakeBuffer({
      0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
  });

  SECTION("Verify input gradient") {
    nnet::Nnet::LearningParameters params{.learning_rate = 0};
    std::unique_ptr<compute::ClBuffer> gradients = test_net.MakeBuffer(0);
    test_net.SetLearningParameters(params);
    test_net.Train(input, expected, gradients);

    gradients->MoveToCpu();
    input->MoveToCpu();

    for (size_t i = 0; i < 9; ++i) {
      std::unique_ptr<compute::ClBuffer> input_left =
          std::make_unique<compute::ClBuffer>(*input);
      input_left->at(i) -= EPSILON;
      std::unique_ptr<compute::ClBuffer> input_right =
          std::make_unique<compute::ClBuffer>(*input);
      input_right->at(i) += EPSILON;
      auto output_left = test_net.Evaluate(input_left);
      auto output_right = test_net.Evaluate(input_right);
      double error_left = test_net.Error(output_left, expected);
      double error_right = test_net.Error(output_right, expected);

      input_left->MoveToCpu();
      input_right->MoveToCpu();
      double approx_gradient = (error_right - error_left) / (2 * EPSILON);
      double actual_gradient = gradients->at(i);

      CAPTURE(i);
      REQUIRE(actual_gradient ==
              Approx(approx_gradient).epsilon(COMPARISON_EPSILON));
    }
  }

  SECTION("Verify weight gradient") {
    ConvSymbolGenerator s({3, 3, 1}, {3, 3, 1, 1, 1, 1});
    nnet::Nnet::LearningParameters params{.learning_rate = 1};

    stats::Normal initializer(0, 5);
    std::vector<double> weight_init_values(
        model.layers[1].weight_buffer().size());
    for (size_t i = 0; i < model.layers[1].weight_buffer().size(); ++i) {
      model.layers[1].weight_buffer()[i] = initializer.sample();
      weight_init_values[i] = model.layers[1].weight_buffer()[i];
    }

    for (size_t i = 0; i < model.layers[1].weight_buffer().size(); ++i) {
      // A neural network with the weight tweaked left.
      model.layers[1].weight_buffer()[i] = weight_init_values[i] - EPSILON;
      Nnet test_net_tweak_left(model, Nnet::NoWeightInit, MeanSquared);

      // I'm using input (registered with test_net) with test_net_tweak_left.
      // Hence the error.
      test_net_tweak_left.RegisterBuffer(input.get());
      test_net_tweak_left.RegisterBuffer(expected.get());
      auto output_left = test_net_tweak_left.Evaluate(input);
      double error_left = test_net_tweak_left.Error(output_left, expected);

      // A neural network with the weight tweaked right.
      model.layers[1].weight_buffer()[i] = weight_init_values[i] + EPSILON;
      Nnet test_net_tweak_right(model, Nnet::NoWeightInit, MeanSquared);

      test_net_tweak_right.RegisterBuffer(input.get());
      test_net_tweak_right.RegisterBuffer(expected.get());
      auto output_right = test_net_tweak_right.Evaluate(input);
      double error_right = test_net_tweak_right.Error(output_right, expected);

      double approx_gradient = (error_right - error_left) / (2 * EPSILON);

      model.layers[1].weight_buffer()[i] = weight_init_values[i];
      Nnet test_net(model, Nnet::NoWeightInit, MeanSquared);

      test_net.RegisterBuffer(input.get());
      test_net.RegisterBuffer(expected.get());
      test_net.SetLearningParameters(params);
      test_net.Train(input, expected);
      double weight_gradient = weight_init_values[i] - test_net.GetWeight(1, i);

      CAPTURE(i);
      CAPTURE(weight_init_values[i]);
      CAPTURE(test_net.GetWeight(1, i));
      REQUIRE(weight_gradient == Approx(approx_gradient).epsilon(EPSILON));

      // These are currently registered to a temporary nnet
      // (test_net_tweak_right) which is about to be deallocated. Move them to
      // the CPU so they can be re-registered in the next iteration.
      input->MoveToCpu();
      expected->MoveToCpu();
    }
  }
}

TEST_CASE("Softmax Layer unit tests", "[softmaxnet]") {
  constexpr double EPSILON = 0.001;

  constexpr size_t kInputSize = 3;
  constexpr size_t kLayerSize = 3;

  Architecture model(kInputSize);
  model.AddSoftmaxLayer(kLayerSize);

  SECTION("Check softmax output") {
    // Error function and weight initialization do not matter as we are running
    // Evaluate() with a single weightless layer.
    Nnet test_net(model, Nnet::NoWeightInit, MeanSquared);
    auto example = test_net.MakeBuffer({-2.85, 0.86, 0.28});
    auto expected = test_net.MakeBuffer({0.0154493515, 0.631, 0.3533874062});
    auto actual = test_net.Evaluate(example);
    expected->MoveToCpu();
    actual->MoveToCpu();
    for (size_t i = 0; i < 3; ++i) {
      REQUIRE(expected->at(i) == Approx(actual->at(i)).epsilon(EPSILON));
    }
  }

  SECTION("Verify input gradient (cross-checked with approximation)") {
    // Use the model to generate a neural network.
    Nnet test_net(model, Nnet::NoWeightInit, CrossEntropy);
    auto expected = test_net.MakeBuffer({0.2, 0.2, 0.2});

    nnet::Nnet::LearningParameters params{.learning_rate = 0};

    std::unique_ptr<compute::ClBuffer> gradients = test_net.MakeBuffer(0);
    auto input = test_net.MakeBuffer({0.1, 0.2, 0.7});
    test_net.SetLearningParameters(params);
    test_net.Train(input, expected, gradients);

    input->MoveToCpu();
    expected->MoveToCpu();
    gradients->MoveToCpu();

    for (size_t i = 0; i < 3; ++i) {
      std::unique_ptr<compute::ClBuffer> input_left =
          std::make_unique<compute::ClBuffer>(*input);
      input_left->at(i) -= EPSILON;
      std::unique_ptr<compute::ClBuffer> input_right =
          std::make_unique<compute::ClBuffer>(*input);
      input_right->at(i) += EPSILON;
      auto output_left = test_net.Evaluate(input_left);
      double error_left = test_net.Error(output_left, expected);
      auto output_right = test_net.Evaluate(input_right);
      double error_right = test_net.Error(output_right, expected);

      double approx_gradient = (error_right - error_left) / (2 * EPSILON);
      double actual_gradient = gradients->at(i);

      CAPTURE(i);
      REQUIRE(actual_gradient == Approx(approx_gradient).epsilon(EPSILON));
    }
  }
}

TEST_CASE("Cifar model gradient test", "[cifar]") {
  constexpr double EPSILON = 0.00001;
  constexpr double COMPARISON_EPSILON = 0.05;
  constexpr size_t kInputSize = 32 * 32 * 3;
  nnet::Architecture model(kInputSize);
  model
      .AddConvolutionLayer(
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
          })
      .AddMaxPoolLayer(
          /* Input size */ nnet::VolumeDimensions{32, 32, 16},
          /* Output size */ nnet::AreaDimensions{16, 16})
      .AddConvolutionLayer(
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
          })
      .AddMaxPoolLayer(
          /* Input size */ nnet::VolumeDimensions{16, 16, 20},
          /* output size */ nnet::AreaDimensions{8, 8})
      .AddConvolutionLayer(
          {
              8,   // width
              8,   // height
              20,  // R,G,B (depth).
          },
          {
              5,   // filter x size.
              5,   // filter y size.
              20,  // filter z depth size.
              1,   // stride.
              2,   // padding.
              20,  // number of filters.
          })
      .AddMaxPoolLayer(/* Input size */ {8, 8, 20},
                       /* output size */ {4, 4})
      // No activation function, the next layer is softmax which functions as an
      // activation function
      .AddDenseLayer(10, symbolic::Identity)
      .AddSoftmaxLayer(10);
  nnet::Nnet test_net(model, nnet::Nnet::Xavier, nnet::CrossEntropy);

  SECTION("Verify cifar input gradient (cross-checked with approximation)") {
    nnet::Nnet::LearningParameters params{.learning_rate = 0};

    stats::Normal initializer(0, 1);
    auto input = test_net.MakeBuffer(kInputSize);
    for (size_t i = 0; i < kInputSize; ++i) {
      input->at(i) = initializer.sample();
    }

    // Training label doesn' matter, we're just testing that gradients work, so
    // make it random.
    auto train_label = test_net.MakeBuffer(10);
    for (size_t i = 0; i < 10; ++i) {
      train_label->at(i) = initializer.sample();
    }

    // Calculate actual gradients.
    std::unique_ptr<compute::ClBuffer> gradients = test_net.MakeBuffer(0);
    test_net.SetLearningParameters(params);
    test_net.Train(input, train_label, gradients);

    input->MoveToCpu();
    train_label->MoveToCpu();
    gradients->MoveToCpu();

    for (size_t i = 0; i < kInputSize; ++i) {
      std::unique_ptr<compute::ClBuffer> input_left =
          std::make_unique<compute::ClBuffer>(*input);
      std::unique_ptr<compute::ClBuffer> input_right =
          std::make_unique<compute::ClBuffer>(*input);

      input_left->at(i) -= EPSILON;
      input_right->at(i) += EPSILON;

      auto output_left = test_net.Evaluate(input_left);
      double error_left = test_net.Error(output_left, train_label);
      auto output_right = test_net.Evaluate(input_right);
      double error_right = test_net.Error(output_right, train_label);

      double approx_gradient = (error_right - error_left) / (2 * EPSILON);

      double actual_gradient = gradients->at(i);

      CAPTURE(i);
      REQUIRE(actual_gradient ==
              Approx(approx_gradient).epsilon(COMPARISON_EPSILON));
    }
  }
}

}  // namespace nnet

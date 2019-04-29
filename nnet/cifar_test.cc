#include "math/geometry/dynamic_matrix.h"
#include "math/nnet/layer_dimensions.h"
#include "math/nnet/nnet.h"
#include "math/symbolic/expression.h"

#include <cassert>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

constexpr size_t kSampleSize = 32 * 32 * 3;
constexpr size_t kOutputSize = 10;
constexpr size_t kNumExamples = 100;
// Each record in the training files is one label byte and kSampleSize sample
// bytes.
constexpr size_t kRecordSize = kSampleSize + 1;

std::string LabelToString(uint8_t label);

struct Sample {
  char label;
  char pixels[kSampleSize];

  explicit Sample(char data[kRecordSize]) {
    label = data[0];
    memcpy(&pixels[0], &data[1], kSampleSize);
  }

  std::string Label() const {
    return LabelToString(label);
  }

  // This isn't one-hot input, it's just a column vector input.
  Matrix<double> OneHotEncodedInput() const {
    Matrix<double> input(kSampleSize, 1, 0);
    double norm = 0;
    for (size_t i = 0; i < kSampleSize; ++i) {
      norm += static_cast<double>(pixels[i]) * pixels[i];
    }
    norm = sqrt(norm);
    if (norm == 0) {
      norm = 1;
    }
    for (size_t i = 0; i < kSampleSize; ++i) {
      input.at(i, 0) = static_cast<double>(pixels[i]) / norm;
    }
    return input;
  }

  Matrix<double> OneHotEncodedOutput() const {
    // Initialize kOutputSizex1 blank column vector.
    Matrix<double> output(kOutputSize, 1, 0);

    // Zero the inputs.
    for (size_t i = 0; i < kOutputSize; ++i) {
      output.at(i, 0) = 0.0;
    }

    output.at(label, 0) = 1.0;

    return output;
  }
};

enum Label : uint8_t {
  AIRPLANE = 0,
  AUTOMOBILE,
  BIRD,
  CAT,
  DEER,
  DOG,
  FROG,
  HORSE,
  SHIP,
  TRUCK
};

std::string LabelToString(uint8_t label) {
  switch (label) {
  case AIRPLANE:
    return "Airplane";
  case AUTOMOBILE:
    return "Automobile";
  case BIRD:
    return "Bird";
  case CAT:
    return "Cat";
  case DEER:
    return "Deer";
  case DOG:
    return "Dog";
  case FROG:
    return "Frog";
  case HORSE:
    return "Horse";
  case SHIP:
    return "Ship";
  case TRUCK:
    return "Truck";
  default:
    return "Unknown?!?";
  }
}

std::string OneHotEncodedOutputToString(Matrix<double> output) {
  uint8_t max_index = 0;
  for (size_t index = 0; index < std::get<0>(output.size()); ++index) {
    if (output.at(index, 0) > output.at(max_index, 0)) {
      max_index = index;
    }
  }

  return LabelToString(max_index);
}

// Trains a neural network to learn if given point is in unit circle.
int main() {
  constexpr int kInputSize = kSampleSize;

  // This model taken from:
  // http://cs231n.github.io/convolutional-networks/
  //
  // This is just for debugging reference. It may become stale if not kept up to
  // date. These are the internal layer numbers created in the model below
  // (output on top, input on bottom).
  //
  // Layer 12 softmax_layer_12
	// Layer 11 activation_layer_11
	// Layer 10 dense_layer_10
	// Layer 9 max_pool_layer_9
	// Layer 8 activation_layer_8
	// Layer 7 convolution_layer_7
	// Layer 6 max_pool_layer_6
	// Layer 5 activation_layer_5
	// Layer 4 convolution_layer_4
	// Layer 3 max_pool_layer_3
	// Layer 2 activation_layer_2
	// Layer 1 convolution_layer_1
	// Layer 0 activation_layer_0

  nnet::Architecture model(kInputSize);
  model
      .AddConvolutionLayer(
          {
              32, // width
              32, // height
              3,  // R,G,B (depth).
          },
          {
              5,  // filter x size.
              5,  // filter y size.
              3,  // filter z depth size.
              1,  // stride.
              2,  // padding.
              16, // number of filters.
          })
      .AddMaxPoolLayer(
          /* Input size */ nnet::VolumeDimensions{32, 32, 16},
          /* Output size */ nnet::AreaDimensions{16, 16})
      .AddConvolutionLayer(
          {
              16, // width
              16, // height
              16, // R,G,B (depth).
          },
          {
              5,  // filter x size.
              5,  // filter y size.
              16, // filter z depth size.
              1,  // stride.
              2,  // padding.
              20, // number of filters.
          })
      .AddMaxPoolLayer(
          /* Input size */ nnet::VolumeDimensions{16, 16, 20},
          /* output size */ nnet::AreaDimensions{8, 8})
      .AddConvolutionLayer(
          {
              8,  // width
              8,  // height
              20, // R,G,B (depth).
          },
          {
              5,  // filter x size.
              5,  // filter y size.
              20, // filter z depth size.
              1,  // stride.
              2,  // padding.
              20, // number of filters.
          })
      .AddMaxPoolLayer(/* Input size */ {8, 8, 20},
                       /* output size */ {4, 4})
      // No activation function, the next layer is softmax which functions as an
      // activation function
      .AddDenseLayer(10, symbolic::Identity)
      .AddSoftmaxLayer(10);
  std::cout << "Initializing network..." << std::endl;
  nnet::Nnet test_net(model, nnet::Nnet::Xavier, nnet::Nnet::CrossEntropy);

  // Read in the files.
  std::vector<string> training_files = {
      "math/nnet/data/cifar-10-batches-bin/data_batch_1.bin",
      "math/nnet/data/cifar-10-batches-bin/data_batch_2.bin",
      "math/nnet/data/cifar-10-batches-bin/data_batch_3.bin",
      "math/nnet/data/cifar-10-batches-bin/data_batch_4.bin",
      "math/nnet/data/cifar-10-batches-bin/data_batch_5.bin",
  };

  std::cout << "Reading in training files..." << std::endl;
  std::vector<Sample> samples;
  for (const string& file : training_files) {
    std::ifstream training_file(file.c_str());
    std::stringstream buffer;
    buffer << training_file.rdbuf();
    std::string sample_buffer = buffer.str();
    std::cout << "Loading file " << file << "..." << std::endl;
    for (size_t sample_index = 0; sample_index < sample_buffer.size();
         sample_index += kRecordSize) {
      if (sample_index + kRecordSize <= sample_buffer.size()) {
        samples.push_back(
            Sample(static_cast<char*>(&sample_buffer[sample_index])));
      }
    }
  }
  std::cout << "Loaded " << samples.size() << " Samples!" << std::endl;

  int samples_so_far = 0;
  double error_sum = 0;
  const size_t kNumTrainingEpochs = 20;
  for (size_t epoch = 1; epoch <= kNumTrainingEpochs; ++epoch) {
    nnet::Nnet::LearningParameters params{.learning_rate = 0.01};
    for (const auto& sample : samples) {
      if (samples_so_far++ % 500 == 0) {
        std::cout << "Progress: " << samples_so_far - 1 << " / " << (kNumTrainingEpochs * samples.size()) << std::endl;
        std::cout << "Error avg over past 500 samples: " << error_sum / 500.0
                  << std::endl;
        std::cout << "Epoch " << epoch << std::endl;
        error_sum = 0;
      }
      test_net.Train(sample.OneHotEncodedInput(), sample.OneHotEncodedOutput(),
                     params);
      error_sum += test_net.Error(sample.OneHotEncodedInput(),
                                  sample.OneHotEncodedOutput());
    }
  }

  std::cout << "Network Weights: " << std::endl;
  std::cout << test_net.WeightsToString() << std::endl;
  std::cout << "Trained over " << samples.size() << " Samples!" << std::endl;

  std::cout << "Example network outputs: " << std::endl;
  srand(time(nullptr));
  int correct_count = 0;
  for (size_t i = 0; i < kNumExamples; ++i) {
    size_t example_index = std::rand() % samples.size();
    std::string actual = samples[example_index].Label();
    std::string nnet_output = OneHotEncodedOutputToString(
        test_net.Evaluate(samples[example_index].OneHotEncodedInput()));
    std::cout << "=================================" << std::endl;
    std::cout << "Actual Answer: " << actual << "\nnnet output: " << nnet_output
              << std::endl;
    if (actual == nnet_output) {
      correct_count++;
    }
  }
  std::cout << "Examples correct: " << correct_count << " / " << kNumExamples
            << " (" << 100.0 * static_cast<double>(correct_count) / kNumExamples
            << "%)" << std::endl;

  std::cout << std::endl;
  return 0;
}

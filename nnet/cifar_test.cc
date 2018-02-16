#include "math/geometry/dynamic_matrix.h"
#include "math/nnet/nnet.h"
#include "math/symbolic/expression.h"

#include <cassert>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

constexpr size_t kSampleSize = 32 * 32 * 3;
constexpr size_t kOutputSize = 10;
// Each record in the training files is one label byte and kSampleSize sample
// bytes.
constexpr size_t kRecordSize = kSampleSize + 1;

struct Sample {
  char label;
  char pixels[kSampleSize];

  explicit Sample(char data[kRecordSize]) {
    label = data[0];
    memcpy(this, &data[1], kSampleSize);
  }

  Matrix<double> OneHotEncodedInput() {
    Matrix<double> input(kSampleSize, 1, 0);
    for (size_t i = 0; i < kSampleSize; ++i) {
      input.at(i, 0) = static_cast<double>(pixels[i]);
    }
    return input;
  }

  Matrix<double> OneHotEncodedOutput() {
    // Initialize kOutputSizex1 blank column vector.
    Matrix<double> output(kOutputSize, 1, 0);

    output.at(label, 0) = 1;

    return output;
  }
};

enum Label {
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

std::string OneHotEncodedOutputToString(Matrix<double> output) {
  size_t max_index = 0;
  for (size_t index = 0; index < std::get<0>(output.size()); ++index) {
    if (output.at(index, 0) > output.at(max_index, 0)) {
      max_index = index;
    }
  }

  switch (max_index) {
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

// Trains a neural network to learn if given point is in unit circle.
int main() {
  constexpr int kInputSize = kSampleSize;

  nnet::Nnet::Architecture model(kInputSize);
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
      .AddActivationLayer(symbolic::Relu)
      .AddMaxPoolLayer(
          /* Input size */ nnet::ConvolutionLayer::VolumeDimensions{32, 32, 16},
          /* output size */ nnet::MaxPoolLayer::AreaDimensions{16, 16})
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
      .AddActivationLayer(symbolic::Relu)
      .AddMaxPoolLayer(
          /* Input size */ nnet::ConvolutionLayer::VolumeDimensions{16, 16, 20},
          /* output size */ nnet::MaxPoolLayer::AreaDimensions{8, 8})
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
      .AddActivationLayer(symbolic::Relu)
      .AddMaxPoolLayer(/* Input size */ {8, 8, 20},
                       /* output size */ {4, 4})
      .AddFeedForwardLayer(10)
      .AddSoftmaxLayer(10);
  nnet::Nnet test_net(model);

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

  // Need to re-code, this is circle test just for reference.
  // std::vector<std::tuple<Sample, bool>> examples;

  //// Generate training samples.
  // for (size_t i = 0; i < 3000; ++i) {
  //  double x = (2.5 * static_cast<double>(std::rand()) / RAND_MAX) - 1.25;
  //  double y = (2.5 * static_cast<double>(std::rand()) / RAND_MAX) - 1.25;

  //  double in_unit_circle = ((x * x + y * y) <= 1.0) ? 1 : 0;

  //  examples.push_back(std::make_tuple(Sample({{x}, {y}}), in_unit_circle));
  //}

  // nnet::Nnet::LearningParameters params{.learning_rate = 1};

  // for (const std::tuple<Sample, double>& example : examples) {
  //  test_net.TrainCl(std::get<0>(example),
  //                   Matrix<nnet::Number>(
  //                       {{static_cast<nnet::Number>(std::get<1>(example))}}),
  //                   params);
  //}

  // std::cout << std::endl;

  // for (size_t i = 0; i < 1000; ++i) {
  //  double pointx = (2.5 * static_cast<double>(std::rand()) / RAND_MAX)
  //  - 1.25; double pointy = (2.5 * static_cast<double>(std::rand()) /
  //  RAND_MAX) - 1.25; double output =
  //      test_net.EvaluateCl(Matrix<nnet::Number>{{pointx}, {pointy}}).at(0,
  //      0);
  //  std::cout << "((" << pointx << "," << pointy << ")," << output << ")"
  //            << std::endl;
  //}

  // std::cout << std::endl;
  // return 0;
}

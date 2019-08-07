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

  std::unique_ptr<memory::ClBuffer> NormalizedInput(nnet::Nnet *network) const {
    std::unique_ptr<memory::ClBuffer> input = network->MakeBuffer(kSampleSize);
    double norm = 0;
    for (size_t i = 0; i < kSampleSize; ++i) {
      norm += static_cast<double>(pixels[i]) * pixels[i];
    }
    norm = sqrt(norm);
    if (norm == 0) {
      norm = 1;
    }
    for (size_t i = 0; i < kSampleSize; ++i) {
      input->at(i) = static_cast<double>(pixels[i]) / norm;
    }
    return input;
  }

  std::unique_ptr<memory::ClBuffer> OneHotEncodedOutput(
      nnet::Nnet* network) const {
    // Initialize kOutputSizex1 blank column vector.
    std::unique_ptr<memory::ClBuffer> output = network->MakeBuffer(kOutputSize);

    // Zero the inputs.
    for (size_t i = 0; i < kOutputSize; ++i) {
      output->at(i) = 0.0;
    }

    output->at(label) = 1.0;

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

std::string OneHotEncodedOutputToString(std::unique_ptr<memory::ClBuffer> buffer) {
  buffer->MoveToCpu();
  uint8_t max_index = 0;
  for (size_t index = 0; index < buffer->size(); ++index) {
    if (buffer->at(index) > buffer->at(max_index)) {
      max_index = index;
    }
  }

  return LabelToString(max_index);
}

void PrintStatus(nnet::Nnet* test_net, const std::vector<Sample>& samples, size_t num_examples) {
  std::cout << "Network Weights: " << std::endl;
  std::cout << test_net->WeightsToString() << std::endl;
  std::cout << "Trained over " << samples.size() << " Samples!" << std::endl;

  std::cout << "Example network outputs: " << std::endl;
  srand(time(nullptr));
  int correct_count = 0;
  for (size_t i = 0; i < num_examples; ++i) {
    size_t random_integer = std::rand();
    size_t example_index = random_integer % samples.size();
    std::string actual = samples[example_index].Label();
    std::string nnet_output = OneHotEncodedOutputToString(
        test_net->Evaluate(samples[example_index].NormalizedInput(test_net)));
    std::cout << "=================================" << std::endl;
    std::cout << "Actual Answer: " << actual << "\nnnet output: " << nnet_output
              << std::endl;
    if (actual == nnet_output) {
      correct_count++;
    }
  }
  std::cout << "Examples correct: " << correct_count << " / " << num_examples
            << " (" << 100.0 * static_cast<double>(correct_count) / num_examples
            << "%)" << std::endl;

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
  nnet::Nnet test_net(model, nnet::Nnet::Xavier, nnet::CrossEntropy);

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
  std::cout << "Loaded " << samples.size() << " Samples from disk!" << std::endl;
  if (samples.size() == 0) {
    std::cerr << "No samples found, quitting!" << std::endl;
    return -1;
  }
  std::cout << "Moving samples to GPU..." << std::endl;
  std::vector<std::unique_ptr<memory::ClBuffer>> inputs;
  std::vector<std::unique_ptr<memory::ClBuffer>> outputs;
  for (const auto& sample : samples) {
      auto input = sample.NormalizedInput(&test_net);
      auto expected = sample.OneHotEncodedOutput(&test_net);
      input->MoveToGpu();
      expected->MoveToGpu();
      inputs.emplace_back(std::move(input));
      outputs.emplace_back(std::move(expected));
  }
  std::cout << "Entire dataset of " << inputs.size()
            << " examples is now stored on GPU!" << std::endl;

  nnet::Nnet::LearningParameters params{.learning_rate = 0.001};
  test_net.SetLearningParameters(params);

  int samples_so_far = 0;
  const size_t kNumTrainingEpochs = 1;
  for (size_t epoch = 1; epoch <= kNumTrainingEpochs; ++epoch) {
    for (size_t i = 0; i < samples.size(); ++i) {
      if (samples_so_far % 100000 == 0) {
        PrintStatus(&test_net, samples, 1000);
      }
      if (samples_so_far++ % 5000 == 0) {
        std::cout << "Progress: " << samples_so_far - 1 << " / " << (kNumTrainingEpochs * samples.size()) << std::endl;
        std::cout << "Epoch " << epoch << std::endl;
      }
      auto& input = inputs[i];
      auto& expected = outputs[i];
      // test_net.Train(input, expected);
      test_net.Evaluate(input);
    }
  }

  std::cout << std::endl;
  return 0;
}

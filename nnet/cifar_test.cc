#include "geometry/dynamic_matrix.h"
#include "nnet/layer_dimensions.h"
#include "nnet/nnet.h"
#include "symbolic/expression.h"
#include "external/libjpeg_turbo/turbojpeg.h"

#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <cstdlib>
#include <sys/stat.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

constexpr size_t kSampleSize = 32 * 32 * 3;
constexpr size_t kOutputSize = 10;
constexpr size_t kNumExamples = 100;
// Each record in the training files is one label byte and kSampleSize sample
// bytes.
constexpr size_t kRecordSize = kSampleSize + 1;

inline constexpr char kSaveDirectory[] = "/home/sharf/cifar10/";

bool file_exists(const std::string &path) {
  std::ifstream f(path.c_str());
  return f.good();
}

std::string CalculateWeightFileName() {
  int frame_index = 1;
  std::string candidate_name;
  do {
    candidate_name = std::string(kSaveDirectory) + "weights_" + std::to_string(frame_index) + ".json";
    frame_index++;
  } while(file_exists(candidate_name));
  return candidate_name;
}

std::string LabelToString(uint8_t label);

struct Jpeg {
  int width;
  int height;
  int subsample;
  int quality;  // 0 to 100.
  uint8_t *data;
  size_t data_size;
};

struct Sample {
  char label;
  char pixels[kSampleSize];

  explicit Sample(char data[kRecordSize]) {
    label = data[0];
    memcpy(&pixels[0], &data[1], kSampleSize);
  }

  std::string Label() const { return LabelToString(label); }

  std::unique_ptr<compute::ClBuffer> NormalizedInput(
      nnet::Nnet* network) const {
    std::unique_ptr<compute::ClBuffer> input = network->MakeBuffer(kSampleSize);
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

  std::unique_ptr<compute::ClBuffer> OneHotEncodedOutput(
      nnet::Nnet* network) const {
    // Initialize kOutputSizex1 blank column vector.
    std::unique_ptr<compute::ClBuffer> output =
        network->MakeBuffer(kOutputSize);

    // Zero the inputs.
    for (size_t i = 0; i < kOutputSize; ++i) {
      output->at(i) = 0.0;
    }

    output->at(label) = 1.0;

    return output;
  }
};

Jpeg encode_jpeg(uint8_t *data, size_t width, size_t height) {
  tjhandle operation = tjInitCompress();
  assert(operation != nullptr);
  int pixelFormat = TJPF_RGB;
  Jpeg jpeg;
  jpeg.subsample = TJSAMP_444;
  jpeg.quality = 100;
  jpeg.data = nullptr;
  if (tjCompress2(operation, data, width, 0, height, pixelFormat, &jpeg.data, &jpeg.data_size, jpeg.subsample, jpeg.quality, 0) < 0) {
    std::cout << "tjCompress2 failed: " << tjGetErrorStr() << std::endl;
    std::exit(0);
  }
  tjDestroy(operation);
  return jpeg;
}

void WriteCifarImageToFile(const std::string &directory, const Sample &sample) {
  uint8_t rgbimage[kSampleSize];
  std::cout << "Dir: " << directory << std::endl;
  for (size_t i = 0; i < 1024; i++) {
    const int32_t rgbimage_offset = i * 3;
    rgbimage[rgbimage_offset] = sample.pixels[i];
    rgbimage[rgbimage_offset + 1] = sample.pixels[1024 + i];
    rgbimage[rgbimage_offset + 2] = sample.pixels[2048 + i];
  }
  Jpeg jpeg = encode_jpeg(rgbimage, 32, 32);
  std::string path = directory + "/XXXXXX.jpg";
  const int fd = mkstemps(path.data(), 4);
  assert(fd >= 0);
  size_t offset = 0;
  while (offset < jpeg.data_size) {
    const int numbytes = write(fd, jpeg.data + offset, jpeg.data_size - offset);
    offset += numbytes;
  }
  close(fd);
  return;
}

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
      return "Unknown";
  }
}

std::string OneHotEncodedOutputToString(
    std::unique_ptr<compute::ClBuffer> buffer) {
  buffer->MoveToCpu();
  uint8_t max_index = 0;
  for (size_t index = 0; index < buffer->size(); ++index) {
    if (buffer->at(index) > buffer->at(max_index)) {
      max_index = index;
    }
  }

  return LabelToString(max_index);
}

void LoadCifarFile(const std::string &file, std::vector<Sample>
*out_samples) {
  std::ifstream training_file(file.c_str());
  std::stringstream buffer;
  buffer << training_file.rdbuf();
  std::string sample_buffer = buffer.str();
  std::cout << "Loading file " << file << "..." << std::endl;
  for (size_t sample_index = 0; sample_index < sample_buffer.size();
       sample_index += kRecordSize) {
    if (sample_index + kRecordSize <= sample_buffer.size()) {
      out_samples->push_back(
          Sample(static_cast<char*>(&sample_buffer[sample_index])));
    }
  }
}

void SaveWeightsToFile(nnet::Nnet* test_net, const std::string &weight_file_path) {
  const std::string weights = test_net->WeightsToString();
  // Save a frame.
  std::ofstream weight_file;
  std::cout << "Saving weights to file: " << weight_file_path << std::endl;
  weight_file.open(weight_file_path, std::ios::out);
  weight_file << test_net->WeightsToString();
  weight_file.close();
}

void PrintStatus(nnet::Nnet* test_net, const std::vector<Sample>& samples,
                 size_t num_examples) {
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
int main(int argc, char *argv[]) {
  // This option exists for profiling/debugging.
  bool only_one_epoch = false;
  std::string weight_file_path = "";
  std::set<std::string> options;
  std::vector<std::string> pos_args;
  for (int i = 1; i < argc; i++) {
    const std::string arg = argv[i];
    if (arg.find("--") == 0) {
      options.insert(arg);
    }
    else {
      pos_args.push_back(arg);
    }
  }
  if(options.count("--short") == 1) {
    only_one_epoch = true;
  }
  if (pos_args.size() >= 1) {
    weight_file_path = pos_args[0];
  }
  const size_t kNumTrainingEpochs = (only_one_epoch) ? 0 : 100000;

  std::cout << "Number of Epochs: " << kNumTrainingEpochs << std::endl;

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
  std::cout << "Initializing network..." << std::endl;
  nnet::Nnet test_net(model, nnet::Nnet::Xavier, nnet::CrossEntropy);

  // Load weights if applicable.
  if (!weight_file_path.empty()) {
    std::cout << "Loading weights from file " << weight_file_path << std::endl;
    std::ifstream weight_file(weight_file_path.c_str());
    std::stringstream buffer;
    buffer << weight_file.rdbuf();
    std::string weight_string = buffer.str();
    if (weight_string.empty()) {
      std::cout << "Provided weight file is empty. Initializing with random weights" << std::endl;
    } else {
      if (!test_net.LoadWeightsFromString(weight_string)) {
        std::cerr << "Failed to load weights from file: " << weight_file_path.c_str() << std::endl;
        return 1;
      }
    }
  }

  // Read in the files.
  std::vector<string> training_files = {
      "nnet/data/cifar-10-batches-bin/data_batch_1.bin",
      "nnet/data/cifar-10-batches-bin/data_batch_2.bin",
      "nnet/data/cifar-10-batches-bin/data_batch_3.bin",
      "nnet/data/cifar-10-batches-bin/data_batch_4.bin",
      "nnet/data/cifar-10-batches-bin/data_batch_5.bin",
  };
  std::string test_file = "nnet/data/cifar-10-batches-bin/test_batch.bin";
  std::vector <Sample> test_batch;
  LoadCifarFile(test_file, &test_batch);

  std::cout << "Reading in training files..." << std::endl;
  std::vector<Sample> samples;
  for (const string& file : training_files) {
    LoadCifarFile(file, &samples);
  }

  std::cout << "Loaded " << samples.size() << " Samples from disk!"
            << std::endl;
  if (samples.size() == 0) {
    std::cerr << "No samples found, quitting!" << std::endl;
    return -1;
  }
  std::cout << "Moving samples to GPU..." << std::endl;
  std::vector<std::unique_ptr<compute::ClBuffer>> inputs;
  std::vector<std::unique_ptr<compute::ClBuffer>> outputs;
  
  for (const auto& sample : samples) {
    auto input = sample.NormalizedInput(&test_net);
    auto expected = sample.OneHotEncodedOutput(&test_net);
    input->MoveToGpu();
    expected->MoveToGpu();
    inputs.emplace_back(std::move(input));
    outputs.emplace_back(std::move(expected));
  }
  for (const auto& sample : test_batch) {
    auto input = sample.NormalizedInput(&test_net);
    auto expected = sample.OneHotEncodedOutput(&test_net);
    input->MoveToGpu();
    expected->MoveToGpu();
    inputs.emplace_back(std::move(input));
    outputs.emplace_back(std::move(expected));
  }
  std::cout << "Entire dataset of " << inputs.size()
            << " examples is now stored on GPU!" << std::endl;

  nnet::Nnet::LearningParameters params{.learning_rate = 0.0001};
  test_net.SetLearningParameters(params);

  // constexpr int kBatchSize=50;

  int samples_so_far = 0;
  auto start = std::chrono::high_resolution_clock::now();
  for (size_t epoch = 1; epoch <= kNumTrainingEpochs; ++epoch) {
    for (size_t i = 0; i < samples.size(); i += 1) {
      auto& input = inputs[i];
      auto& expected = outputs[i];
      // SGD
      test_net.Train(input, expected);
      
      // Minibatch.
      // std::vector<int> batch(kBatchSize);
      // std::generate(batch.begin(), batch.end(), std::rand);
      // std::set<int> batch_set(batch.begin(), batch.end());
      // test_net.BatchTrain(inputs, outputs, batch_set);

      samples_so_far++;
      if (samples_so_far % 100000 == 0) {
        PrintStatus(&test_net, samples, 1000);
        SaveWeightsToFile(&test_net, weight_file_path);
      }
      if (samples_so_far % 5000 == 0) {
        std::cout << "Progress: " << samples_so_far - 1 << " / "
                  << (kNumTrainingEpochs * samples.size()) << std::endl;
        std::cout << "Epoch " << epoch << std::endl;
        auto end = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        double rate = 1000 * 5000.0 / duration.count();
        std::cout << "Training rate (samples per second): " << rate
                  << std::endl;
        start = std::chrono::high_resolution_clock::now();
      }
    }
  }

  std::cout << "Training completed!" << std::endl;

  char kTempDirectory[] = "/tmp/cifarsort.XXXXXX";
  std::string tempdir(mkdtemp(kTempDirectory));
  std::cout << "Sorting & outputing files to: " << tempdir << std::endl;
  assert(0 == mkdir((tempdir + "/Airplane").c_str(), 0766));
  assert(0 == mkdir((tempdir + "/Automobile").c_str(), 0766));
  assert(0 == mkdir((tempdir + "/Bird").c_str(), 0766));
  assert(0 == mkdir((tempdir + "/Cat").c_str(), 0766));
  assert(0 == mkdir((tempdir + "/Deer").c_str(), 0766));
  assert(0 == mkdir((tempdir + "/Dog").c_str(), 0766));
  assert(0 == mkdir((tempdir + "/Frog").c_str(), 0766));
  assert(0 == mkdir((tempdir + "/Horse").c_str(), 0766));
  assert(0 == mkdir((tempdir + "/Ship").c_str(), 0766));
  assert(0 == mkdir((tempdir + "/Truck").c_str(), 0766));
  assert(0 == mkdir((tempdir + "/Unknown").c_str(), 0766));
  std::cout << "samples: " << test_batch.size() << std::endl;
  for (size_t i = 0; i < test_batch.size(); ++i) {
    std::string nnet_output = OneHotEncodedOutputToString(
        test_net.Evaluate(test_batch[i].NormalizedInput(&test_net)));
    std::string directory = (tempdir + "/" + nnet_output);
    std::cout << "Saving to: " << directory << std::endl; 
    WriteCifarImageToFile(directory, test_batch[i]);
  }
  return 0;
}

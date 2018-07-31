#ifndef ARCHITECTURE_H
#define ARCHITECTURE_H

#include "math/nnet/layer.h"
#include "math/nnet/layer_dimensions.h"
#include "math/symbolic/expression.h"
#include "math/symbolic/symbolic_util.h"

#include <vector>

namespace nnet {

struct Architecture {
  std::vector<Layer> layers;

  bool VerifyArchitecture() const {
    // Cannot be an empty architecture.
    if (layers.size() == 0) {
      return false;
    }
    for (size_t i = 1; i < layers.size(); ++i) {
      size_t prev_output = layers[i - 1].GetDimensions().num_outputs;
      size_t curr_input = layers[i].GetDimensions().num_inputs;
      if (prev_output != curr_input) {
        return false;
      }
    }
    return true;
  }

  Architecture(size_t input_size) { AddInputLayer(input_size); }

  Architecture() {}

  // TODO(sharf): move all these types into layer_types.h to mirror
  // layer_dimensions.h.
  // std::function<symbolic::Expression(const symbolic::Expression&)>
  using ActivationFunctionType = Layer::ActivationFunctionType;

  // Input layer is just an activation layer with zero activation. Used for
  // semantics and to specify input size.
  Architecture& AddInputLayer(size_t size) {
    return AddActivationLayer(size, symbolic::Identity);
  }

  Architecture& AddDenseLayer(
      size_t num_outputs, const ActivationFunctionType& activation_function) {
    Dimensions dimensions = {
        // Num inputs = num previous layer outputs.
        layers[layers.size() - 1].GetDimensions().num_outputs,
        // Num outputs specified by input parameter.
        num_outputs,
    };

    layers.push_back(Layer::MakeDenseLayer(layers.size(), dimensions,
                                                 activation_function));
    return *this;
  }

  Architecture& AddDenseLayer(size_t num_outputs) {
    Dimensions dimensions = {
        // Num inputs = num previous layer outputs.
        layers[layers.size() - 1].GetDimensions().num_outputs,
        // Num outputs specified by input parameter.
        num_outputs,
    };

    layers.push_back(Layer::MakeDenseLayer(layers.size(), dimensions));
    return *this;
  }

  Architecture& AddConvolutionLayer(const VolumeDimensions& dimensions,
                                    const FilterParams& params) {
    layers.push_back(
        Layer::MakeConvolutionLayer(layers.size(), dimensions, params));
    return *this;
  }

  Architecture& AddSoftmaxLayer(size_t size) {
    layers.push_back(Layer::MakeSoftmaxLayer(layers.size(), size));
    return *this;
  }

  Architecture& AddActivationLayer(
      size_t size, const ActivationFunctionType& activation_function) {
    layers.push_back(
        Layer::MakeActivationLayer(layers.size(), size, activation_function));
    return *this;
  }

  Architecture& AddActivationLayer(
      const ActivationFunctionType& activation_function) {
    return AddActivationLayer(
        layers[layers.size() - 1].GetDimensions().num_outputs,
        activation_function);
  }

  Architecture& AddMaxPoolLayer(const VolumeDimensions& input,
                                const AreaDimensions& output) {
    layers.push_back(Layer::MakeMaxPoolLayer(layers.size(), input, output));
    return *this;
  }

  std::string to_string() const {
    std::stringstream buffer;
    buffer << "Architecture{ inputs: " << input_size()
           << ", outputs: " << output_size() << "}";
    return buffer.str();
  }

  size_t input_size() const { return layers[0].GetDimensions().num_inputs; }
  size_t output_size() const {
    return layers[layers.size() - 1].GetDimensions().num_outputs;
  }
};

}  // namespace nnet

#endif  // ARCHITECHTURE_H

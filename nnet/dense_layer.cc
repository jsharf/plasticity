#include "math/nnet/dense_layer.h"

#include <cassert>

namespace nnet {

DenseLayer::DenseLayer(const Dimensions& dimensions,
                       const ActivationFunctionType& activation_function,
                       size_t layer_index)
    : Super(dimensions, layer_index),
      generator_(dimensions),
      dimensions_(dimensions),
      activation_function_(activation_function) {}

const std::vector<std::string>& DenseLayer::weights() const {
  return generator_.weights();
}

void DenseLayer::GenerateOutputCode(const symbolic::Expression &output_index,
                                    codegen::Generator *cg) const {
  symbolic::Expression sum = 0;
  for (size_t i = 0; i < dimensions_.num_inputs; ++i) {
    sum += generator_.W(output_index, Expression(i)) * generator_.I(i);
  }

  // Bias input.
  sum += generator_.W(output_index) * 1;

  symbolic::Expression retval = activation_function_(sum);
  cg->AppendLineOfCode("return " + retval.to_string() + cg->linesep());
}

// The input gradient is just the sum of the weights between each output and that
// input multiplied by the back-propagated gradients.
void DenseLayer::InputGradientCode(const symbolic::Expression &input_index,
                                   codegen::Generator *cg) const {
  symbolic::Expression sum = 0;
  for (size_t out_index = 0; out_index < dimensions_.num_outputs; ++out_index) {
    sum += generator_.GRADIENT(out_index) * generator_.W(symbolic::Expression(out_index), input_index);
  }
  cg->AppendLineOfCode("return " + sum.to_string() + cg->linesep());
}

// The weight gradient is just the input for that weight multiplied by the
// back-propagated gradient.
void DenseLayer::WeightGradientCode(
    const symbolic::Expression& weight_index, codegen::Generator* cg) const {
  // Unflatten the weight index to node, edge.
  symbolic::Expression node = symbolic::Unflatten2dRow(dimensions_.num_inputs + 1, dimensions_.num_outputs, weight_index);
  symbolic::Expression edge = symbolic::Unflatten2dCol(dimensions_.num_inputs + 1, dimensions_.num_outputs, weight_index);
  symbolic::Expression retval =
      symbolic::IfInRange(edge, 0, dimensions_.num_inputs,
                          generator_.GRADIENT(node) * generator_.I(edge),
                          generator_.GRADIENT(node));
  cg->AppendLineOfCode("return " + retval.to_string() + cg->linesep());
}

std::unique_ptr<LayerImpl> DenseLayer::Clone() const {
  return std::make_unique<DenseLayer>(dimensions_, activation_function_,
                                      layer_index_);
}

}  // namespace nnet

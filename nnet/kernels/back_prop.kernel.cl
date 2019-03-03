// THIS FILE IS A TEMPLATE. The files in //math/nnet parse this to generate an
// OpenCl kernel.

// Backprop weight update stage
// @param GRADIENT: de/do for all outputs of this layer. Calculated by CalculateInputGradient_*
// @param I: inputs
// @param W: weights
// @param weight_index: Which weight to calculate the gradient for.
// output: adjustment for the requested weight_index.
double CalculateWeightGradient_LAYERID(global double* I, global double* W,
                                       const global double* GRADIENT,
                                       int weight_index) {
  WEIGHT_GRADIENTS_HERE
}

// Backprop backwards gradient propagation stage
// @param GRADIENT: de/do for all outputs of this layer. Calculated by CalculateInputGradient_*
// @param I: inputs
// @param W: weights
// @param input_index: Which input to calculate the backwards GRADIENT for.
// output: gradient for the requested input_index.
double CalculateInputGradient_LAYERID(global double* I, global double* W,
                                      const global double* GRADIENT,
                                      int input_index) {
  INPUT_GRADIENTS_HERE
}

kernel void weight_delta_LAYERID(
    global double* inputs, global double* weights,
    const global double* output_gradient,  // back-propagated output gradient.
    global double* new_weights, global double* learning_rate) {
  size_t i = get_global_id(0);
  new_weights[i] =
      weights[i] - learning_rate[0] * CalculateWeightGradient_LAYERID(
                                          inputs, weights, output_gradient, i);
}

kernel void input_delta_LAYERID(
    global double* inputs, global double* weights,
    const global double* output_gradient,  // back-propagated output gradient.
    global double* input_deltas) {
  size_t i = get_global_id(0);
  // de/di = de/do * do/di
  input_deltas[i] =
      CalculateInputGradient_LAYERID(inputs, weights, output_gradient, i);
}

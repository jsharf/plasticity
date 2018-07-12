// THIS FILE IS A TEMPLATE. The files in //math/nnet parse this to generate an
// OpenCl kernel.

// a(i*w) = o
// de/dw = de/do * do/dw
// To gen de/do, must sum up de/di for all inputs in next layer connected to
// this output. Do this in separate Collator kernel.
// do/dw = autocalculated from layer_output[i].Derive("w")
// do/di = autocalculated from layer_output[i].Derive("i")
// find de/dw and generate de/di.

// Backprop gradient calculator.
// inputs: de/do or dE, for all node n outputs.
// inputs: o(i) s.t.
// outputs: de/dw (or do/dw) and de/di for all w,i in node n for all nodes n in
// layer l. Weight updates are 1D grid of size L^2 (layer size squared). de/di
// is 2D LxL (L = layer size) grid that needs to be collated.

// This function is generated automatically, do not edit.
double CalculateWeightGradient_LAYERID(global double* I, global double* W,
                                       const global double* GRADIENT,
                                       size_t weight_index) {
  switch (weight_index) { WEIGHT_GRADIENTS_HERE }
  return NAN;
}

// This function is generated automatically, do not edit.
double CalculateInputGradient_LAYERID(global double* I, global double* W,
                                      const global double* GRADIENT,
                                      size_t input_index) {
  switch (input_index) { INPUT_GRADIENTS_HERE };
  return NAN;
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

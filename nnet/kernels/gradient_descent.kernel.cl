// THIS FILE IS A TEMPLATE. The files in //math/nnet parse this to generate an OpenCl kernel.

// This function is generated automatically, do not edit.
double CalculateDelta(global double* I, global double* W, global double* O,
                     size_t weight_index) {
  switch(weight_index) {
    GRADIENTS_HERE
  }
  return NAN;
}

kernel void gradient_descent(global double* inputs, global double* weights,
global double* expected_results, global double* new_weights, global double* learning_rate) {
  size_t i = get_global_id(0);
  new_weights[i] = weights[i] - learning_rate[0] * CalculateDelta(inputs, weights, expected_results, i);
}

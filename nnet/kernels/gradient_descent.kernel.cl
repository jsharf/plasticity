// THIS FILE IS A TEMPLATE. The files in //math/nnet parse this to generate an OpenCl kernel.

// This function is generated automatically, do not edit.
float CalculateDelta(global float* I, global float* W, global float* O,
                     size_t weight_index) {
  return GRADIENTS_HERE[weight_index];
}

kernel void gradient_descent(global float* inputs, global float* weights, global float* expected_results, float learning_rate) {
  size_t index = get_index();
  weights[i] += learning_rate * CalculateDelta(inputs, weights, expected_results, index);
}

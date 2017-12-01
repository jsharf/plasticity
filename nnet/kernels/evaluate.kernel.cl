// THIS FILE IS A TEMPLATE. The files in //math/nnet parse this to generate an OpenCl kernel.

// This function is generated automatically, do not edit.
void Calculate(global float* I, global float* W, global float* O, size_t output_index) {
  O[output_index] = EXPRESSION_HERE[output_index];
}

kernel void evaluate(global float* inputs, global float* weights, global float* outputs) {
  size_t index = get_index();
  Calculate(inputs, weights, outputs, index);
}

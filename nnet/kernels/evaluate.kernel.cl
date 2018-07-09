// THIS FILE IS A TEMPLATE. The files in //math/nnet parse this to generate an OpenCl kernel.

// This function is generated automatically, do not edit.
double Calculate_LAYERID(global double* I, global double* W, size_t output_index) {
  switch(output_index) {
    EXPRESSION_HERE
  }
  return NAN;
}

kernel void evaluate_LAYERID(global double* inputs, global double* weights, global double* outputs) {
  size_t index = get_global_id(0);
  outputs[index] = Calculate_LAYERID(inputs, weights, index);
}

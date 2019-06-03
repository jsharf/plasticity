kernel double CalculateError(const global double* O, const global double* E, int index) {
  return ERROR_EXPRESSION_HERE;
}

kernel double CalculateErrorGradient(const global double* O, const global double* E, int index) {
  return GRADIENT_EXPRESSION_HERE;
}

kernel void error(
    const global double* output, const global double* expected,
    global double* error_components) {
  size_t i = get_global_id(0);
  error_components[i] = CalculateError(output, expected, i);
}

kernel void error_gradients(
    const global double* output, const global double* expected,
    global double* output_gradient) {
  size_t i = get_global_id(0);
  output_gradient[i] = CalculateErrorGradient(output, expected, i);
}

// a = a + b
kernel void vector_accumulate(global double* a, global double* b) {
  size_t i = get_global_id(0);
  a[i] += b[i];
}

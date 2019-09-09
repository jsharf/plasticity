#define FPTYPE Number;

// a = a + b
kernel void vector_accumulate(global Number* a, global Number* b) {
  size_t i = get_global_id(0);
  a[i] += b[i];
}

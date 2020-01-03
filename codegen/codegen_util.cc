#include "plasticity/codegen/codegen_util.h"

namespace codegen {

// (a <= index && index < b) ? then : ifnot.
std::string IfInRange(std::string index, std::string a, std::string b,
                      std::string then, std::string ifnot) {
  CudaGenerator cgen;
  return cgen.ternary(cgen.op_and(cgen.gte(index, a), cgen.lt(index, b)), then, ifnot);
}

std::string BoundedArrayAccess(std::string array, std::string index,
                               std::string length) {
  CudaGenerator cgen;
  return IfInRange(index, "0", length, cgen.array_access(array, index), "0");
}

// Row-order flattening.
std::string Flatten2d(std::string width, std::string height, std::string row,
                      std::string col) {
  CudaGenerator cgen;
  return cgen.add(cgen.mul(row, width), col);
}

// Assumes row-order flattening.
std::string Unflatten2dRow(std::string width, std::string height,
                           std::string i) {
  CudaGenerator cgen;
  return cgen.div(i, width);
}

// Assumes row-order flattening.
std::string Unflatten2dCol(std::string width, std::string height,
                           std::string i) {
  CudaGenerator cgen;
  return cgen.mod(i, width);
}

}  // namespace codegen

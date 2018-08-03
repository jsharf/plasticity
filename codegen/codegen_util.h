#ifndef CODEGEN_CODEGEN_UTIL_H
#define CODEGEN_CODEGEN_UTIL_H

namespace codegen {

std::string BoundedArrayAccess(std::string array, std::string index, std::string length) {
  CodeGenerator cgen;
  return cgen.ternary(cgen.op_and(cgen.gte(index, "0"), cgen.lt(length)), cgen.array_access(array, index), "0");
}

}  // namespace codegen

#endif  // CODEGEN_CODEGEN_UTIL_H

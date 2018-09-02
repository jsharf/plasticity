#include "math/codegen/codegen.h"

namespace codegen {

// Control code scope.
void CudaGenerator::PushScope() {
  code_ << "{";
  indent_level_++;
}

void CudaGenerator::PopScope() {
  code_ << "}";
  indent_level_--;
}

// The following functions are all const and do not modify the state of the
// object itself. Their functionality is described with the equivalent C++
// syntax.
// lhs = rhs;
std::string CudaGenerator::assign(const std::string& lhs,
                                  const std::string& rhs) const {
  return lhs + "=" + rhs;
}

// lhs += rhs;
std::string CudaGenerator::add_assign(const std::string& lhs,
                                      const std::string& rhs) const {
  return lhs + "+=" + rhs;
}

// value[index];
std::string CudaGenerator::array_access(const std::string& value,
                                        const std::string& index) const {
  return value + "[" + index + "]";
}

// value[index1][index2];
std::string CudaGenerator::array_access_2d(const std::string& value,
                                           const std::string& index1,
                                           const std::string& index2) const {
  return value + "[" + index1 + "][" + index2 + "]";
}

// if(term)
std::string CudaGenerator::if_expr(const std::string& term) const {
  return "if(" + term + ")";
}

// (condition) ? a : b
std::string CudaGenerator::ternary(const std::string& condition,
                                   const std::string& a,
                                   const std::string& b) const {
  return "((" + condition + ") ? " + a + ":" + b + ")";
}

// lhs && rhs; "and" is a reserved keyword so this was named op_and.
std::string CudaGenerator::op_and(const std::string& lhs,
                                  const std::string& rhs) const {
  return lhs + "&&" + rhs;
}

// lhs || rhs; "or" is a reserved keyword so this was named op_or.
std::string CudaGenerator::op_or(const std::string& lhs,
                                 const std::string& rhs) const {
  return lhs + "||" + rhs;
}

// lhs == rhs;
std::string CudaGenerator::equals(const std::string& lhs,
                                  const std::string& rhs) const {
  return lhs + "==" + rhs;
}

// lhs < rhs;
std::string CudaGenerator::lt(const std::string& lhs,
                              const std::string& rhs) const {
  return lhs + "<" + rhs;
}

// lhs > rhs;
std::string CudaGenerator::gt(const std::string& lhs,
                              const std::string& rhs) const {
  return lhs + ">" + rhs;
}

// lhs <= rhs;
std::string CudaGenerator::lte(const std::string& lhs,
                               const std::string& rhs) const {
  return lhs + "<=" + rhs;
}

// lhs >= rhs;
std::string CudaGenerator::gte(const std::string& lhs,
                               const std::string& rhs) const {
  return lhs + ">=" + rhs;
}

// lhs + rhs;
std::string CudaGenerator::add(const std::string& lhs,
                               const std::string& rhs) const {
  return lhs + "+" + rhs;
}

// lhs * rhs;
std::string CudaGenerator::mul(const std::string& lhs,
                               const std::string& rhs) const {
  return lhs + "*" + rhs;
}

// lhs / rhs;
std::string CudaGenerator::mul(const std::string& lhs,
                               const std::string& rhs) const {
  return lhs + "/" + rhs;
}

std::string CudaGenerator::mod(const std::string& lhs,
                               const std::string& rhs) const {
  return lhs + "%" + rhs;
}

std::string CudaGenerator::else_expr() const { return "else"; }

std::string CudaGenerator::linesep() const { return ";"; }

}  // namespace codegen

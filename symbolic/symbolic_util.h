#ifndef SYMBOLIC_UTIL_H
#define SYMBOLIC_UTIL_H

#include "math/symbolic/expression.h"
#include "math/symbolic/numeric_value.h"

namespace symbolic {

Expression Sigmoid(const Expression& a);

Expression Relu(const Expression& a);

Expression Identity(const Expression& a);

Expression Exp(const Expression& a);

Expression SoftMax(const std::vector<Expression>& exprs, int index);

Expression Max(const std::vector<Expression>& exprs);

}  // namespace symbolic

#endif /* SYMBOLIC_UTIL_H */

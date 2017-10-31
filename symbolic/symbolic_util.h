#ifndef SYMBOLIC_UTIL_H
#define SYMBOLIC_UTIL_H

#include "math/symbolic/expression.h"
#include "math/symbolic/numeric_value.h"

#include <memory>

namespace symbolic {
Expression Sigmoid(Expression a) {
  return CreateExpression("1") /
         (CreateExpression("1") +
          ExponentExpression(NumericValue::e, CreateExpression("-1 * x")));
}
}  // namespace symbolic

#endif /* SYMBOLIC_UTIL_H */

#ifndef SYMBOLIC_UTIL_H
#define SYMBOLIC_UTIL_H

#include "math/symbolic/expression.h"
#include "math/symbolic/numeric_value.h"

#include <memory>

namespace symbolic {

Expression Sigmoid(Expression a) {
  return CreateExpression("1") /
         (CreateExpression("1") +
          Expression(std::make_unique<ExponentExpression>(
              NumericValue::e, (CreateExpression("-1") * a).Release())));
}

}  // namespace symbolic

#endif /* SYMBOLIC_UTIL_H */

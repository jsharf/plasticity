#ifndef SYMBOLIC_UTIL_H
#define SYMBOLIC_UTIL_H

#include "math/symbolic/expression.h"
#include "math/symbolic/numeric_value.h"

#include <memory>

namespace symbolic {

Expression Sigmoid(const Expression& a) {
  return CreateExpression("1") /
         (CreateExpression("1") +
          Expression(std::make_unique<ExponentExpression>(
              NumericValue::e, (CreateExpression("-1") * a).GetPointer())));
}

Expression Relu(const Expression& a) {
  return Expression(std::make_unique<IfExpression>(
      std::make_unique<GteExpression>(a.GetPointer(),
                                      CreateExpression("0").Release()),
      a.GetPointer(), CreateExpression("0").Release()));
}

Expression Identity(const Expression& a) { return a; }

namespace internal {

Expression maxexpr(Expression a, std::vector<Expression> exprs) {
  if (exprs.size() == 0) {
    std::cerr << "maxexpr called with empty vec!" << std::endl;
    std::exit(1);
  }
  std::unique_ptr<const ExpressionNode> condexpr =
      std::make_unique<const GteExpression>(a.GetPointer(),
                                            exprs[0].GetPointer());
  for (size_t i = 1; i < exprs.size(); ++i) {
    condexpr = std::make_unique<const AndExpression>(
        std::move(condexpr), std::make_unique<const GteExpression>(
                                 a.GetPointer(), exprs[i].GetPointer()));
  }
  return Expression(std::move(condexpr));
}

}  // namespace internal

Expression Max(const std::vector<Expression>& exprs) {
  if (exprs.size() == 0) {
    std::cerr << "maxexpr called with empty vec!" << std::endl;
    std::exit(1);
  }
  std::unique_ptr<const ExpressionNode> maxstatement =
      exprs[exprs.size() - 1].GetPointer()->Clone();
  for (size_t i = 0; i < exprs.size() - 1; ++i) {
    // Make copy of exprs that does not contain i-expr.
    std::vector<Expression> others = exprs;
    others.erase(others.begin() + i);

    // Make conditional that i-expr is max.
    Expression conditional = internal::maxexpr(exprs[i], others);

    maxstatement = std::make_unique<const IfExpression>(
        conditional.GetPointer(), exprs[i].GetPointer(),
        std::move(maxstatement));
  }
  return Expression(std::move(maxstatement));
}

}  // namespace symbolic

#endif /* SYMBOLIC_UTIL_H */

#include "math/symbolic/symbolic_util.h"

#include "math/symbolic/expression.h"
#include "math/symbolic/numeric_value.h"

#include <memory>

namespace symbolic {

Expression Sigmoid(const Expression& a) {
  return CreateExpression("1") /
         (CreateExpression("1") +
          Expression(std::make_shared<ExponentExpression>(
              NumericValue::e, (CreateExpression("-1") * a).GetPointer())));
}

Expression Relu(const Expression& a) {
  return Expression(std::make_shared<IfExpression>(
      Expression(std::make_shared<GteExpression>(a, CreateExpression("0"))), a,
      CreateExpression("0")));
}

Expression Identity(const Expression& a) { return a; }

Expression Exp(const Expression& a) {
  return Expression(
      std::make_shared<ExponentExpression>(NumericValue::e, a.GetPointer()));
}

Expression Softmax(const Matrix<Expression>& column_vector, int index) {
  Expression expsum = CreateExpression("0");

  size_t height = std::get<0>(column_vector.size());
  for (size_t i = 0; i < height; ++i) {
    expsum = expsum + Exp(column_vector.at(i, 0));
  }
  return Exp(column_vector.at(index, 0)) / expsum;
}

namespace internal {

Expression maxexpr(Expression a, std::vector<Expression> exprs) {
  if (exprs.size() == 0) {
    std::cerr << "maxexpr called with empty vec!" << std::endl;
    std::exit(1);
  }
  std::shared_ptr<const ExpressionNode> condexpr =
      std::make_shared<const GteExpression>(a, exprs[0]);
  for (size_t i = 1; i < exprs.size(); ++i) {
    condexpr = std::make_shared<const AndExpression>(
        condexpr,
        Expression(std::make_shared<const GteExpression>(a, exprs[i])));
  }
  return Expression(std::move(condexpr));
}

}  // namespace internal

Expression Max(const std::vector<Expression>& exprs) {
  if (exprs.size() == 0) {
    std::cerr << "maxexpr called with empty vec!" << std::endl;
    std::exit(1);
  }
  std::shared_ptr<const ExpressionNode> maxstatement =
      exprs[exprs.size() - 1].GetPointer()->Clone();
  for (size_t i = 0; i < exprs.size() - 1; ++i) {
    // Make copy of exprs that does not contain i-expr.
    std::vector<Expression> others = exprs;
    others.erase(others.begin() + i);

    // Make conditional that i-expr is max.
    Expression conditional = internal::maxexpr(exprs[i], others);

    maxstatement = std::make_shared<const IfExpression>(
        conditional.GetPointer(), exprs[i].GetPointer(),
        std::move(maxstatement));
  }
  return Expression(std::move(maxstatement));
}

// Evaluates a matrix of symbolics given an execution environment and returns
// a matrix of real values.
Matrix<Number> MapBindAndEvaluate(Matrix<symbolic::Expression> symbols,
                                  symbolic::Environment env) {
  // Turns symbolic expressions into real numbers.
  std::function<Number(const symbolic::Expression& e)> real_evaluator =
      [&env, &symbols](const symbolic::Expression& e) -> Number {
    auto maybe_value = e.Bind(env).Evaluate();
    if (!maybe_value) {
      // Shit.
      std::cerr << "Well, fuck, not sure how this happened" << std::endl;
      std::cerr << "Failed to evaluate this expression: \n\t"
                << symbols.to_string() << "\nWith environment: \n";
      for (const auto& val : env) {
        std::cerr << "\t" << val.first << ": " << val.second.to_string()
                  << std::endl;
      }
      std::exit(1);
    }
    return maybe_value->real();
  };
  return symbols.Map(real_evaluator);
}

}  // namespace symbolic

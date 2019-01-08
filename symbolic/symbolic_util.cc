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

Expression Log(NumericValue base, const Expression& exp) {
  return std::static_pointer_cast<const ExpressionNode>(
      std::make_shared<const LogExpression>(base, exp));
}

Expression Log(const Expression& exp) {
  return std::static_pointer_cast<const ExpressionNode>(
      std::make_shared<const LogExpression>(NumericValue::e, exp));
}

Expression Exp(NumericValue base, const Expression& exp) {
  return std::static_pointer_cast<const ExpressionNode>(
      std::make_shared<const ExponentExpression>(base, exp));
}

Expression Exp(const Expression& exp) {
  return std::static_pointer_cast<const ExpressionNode>(
      std::make_shared<const ExponentExpression>(NumericValue::e, exp));
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

Expression maxexpr(Expression a, std::vector<Expression> exprs, size_t skip_index) {
  if (exprs.size() <= 1) {
    std::cerr << "internal::maxexpr called with too small vec!" << std::endl;
    std::exit(1);
  }
  // Initialize condexpr with a "truthy" value.
  std::shared_ptr<const ExpressionNode> condexpr =
      symbolic::Expression(1).GetPointer();
  for (size_t i = 0; i < exprs.size(); ++i) {
    if (i == skip_index) {
      continue;
    }
    condexpr = std::make_shared<const AndExpression>(
        condexpr,
        Expression(std::make_shared<GteExpression>(a, exprs[i])));
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
  // This algorithm can probably be made smaller (in terms of the tree size of
  // the output expression).
  for (size_t i = 0; i < exprs.size() - 1; ++i) {
    // Make conditional that i-expr is max.
    Expression conditional = internal::maxexpr(exprs[i], exprs, i);

    maxstatement = std::make_shared<const IfExpression>(
        conditional.GetPointer(), exprs[i].GetPointer(),
        std::move(maxstatement));
  }
  return Expression(std::move(maxstatement));
}

// Evaluates a matrix of symbolics given an execution environment and returns
// a matrix of real values.
Matrix<double> MapBindAndEvaluate(Matrix<symbolic::Expression> symbols,
                                  symbolic::Environment env) {
  // Turns symbolic expressions into real numbers.
  std::function<double(const symbolic::Expression& e)> real_evaluator =
      [&env, &symbols](const symbolic::Expression& e) -> double {
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

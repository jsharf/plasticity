#include "math/symbolic/symbolic_util.h"

#include "math/symbolic/expression.h"
#include "math/symbolic/numeric_value.h"

#include <memory>

namespace symbolic {

Expression Sigmoid(const Expression& a) {
  return Expression(1.0) /
         (Expression(1.0) + Expression(std::make_shared<ExponentExpression>(
                                NumericValue::e, (Expression(-1.0) * a))));
}

Expression Relu(const Expression& a) {
  return Expression(std::make_shared<IfExpression>(
      Expression(std::make_shared<GteExpression>(a, CreateExpression("0"))), a,
      CreateExpression("0")));
}

Expression LeakyRelu(const Expression& a) {
  return Expression(std::make_shared<IfExpression>(
      Expression(std::make_shared<GteExpression>(a, CreateExpression("0"))), a,
      a/10));
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

Expression SafeLog(const Expression& exp) {
  return Log(exp + std::numeric_limits<double>::epsilon());
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
      std::cerr << "Required symbols: " << std::endl;
      for (const auto& val : e.variables()) {
        std::cerr << ">\t" << val << std::endl;
      }
      std::exit(1);
    }
    return maybe_value->real();
  };
  return symbols.Map(real_evaluator);
}

// 3D array flattening & unflattening.

symbolic::Expression Flatten3d(size_t width, size_t height, size_t depth,
                               const symbolic::Expression& row,
                               const symbolic::Expression& col,
                               const symbolic::Expression& plane) {
  symbolic::Expression z_plane_size(symbolic::Integer(width * height));
  return z_plane_size * plane + row * width + col;
}

symbolic::Expression Unflatten3dRow(size_t width, size_t height, size_t depth,
                                    const symbolic::Expression& i) {
  symbolic::Expression z_plane_size(symbolic::Integer(width * height));
  symbolic::Expression z_plane = i / z_plane_size;
  symbolic::Expression plane_index = i - z_plane * z_plane_size;
  symbolic::Expression row = plane_index / width;
  return row;
}

symbolic::Expression Unflatten3dCol(size_t width, size_t height, size_t depth,
                                    const symbolic::Expression& i) {
  symbolic::Expression z_plane_size(symbolic::Integer(width * height));
  symbolic::Expression z_plane = i / z_plane_size;
  symbolic::Expression plane_index = i - z_plane * z_plane_size;
  symbolic::Expression col = plane_index % width;
  return col;
}

symbolic::Expression Unflatten3dPlane(size_t width, size_t height, size_t depth,
                                      const symbolic::Expression& i) {
  symbolic::Expression z_plane_size(symbolic::Integer(width * height));
  symbolic::Expression z_plane = i / z_plane_size;
  return z_plane;
}

// 2D array flattening & unflattening.

symbolic::Expression Flatten2d(size_t width, size_t height,
                               const symbolic::Expression& row,
                               const symbolic::Expression& col) {
  return symbolic::Expression(width) * row + col;
}

symbolic::Expression Unflatten2dRow(size_t width, size_t height,
                                    const symbolic::Expression& i) {
  return i / symbolic::Integer(width);
}

symbolic::Expression Unflatten2dCol(size_t width, size_t height,
                                    const symbolic::Expression& i) {
  return i % symbolic::Integer(width);
}

// LT expression
symbolic::Expression LtExpression(const symbolic::Expression& a,
                                  const symbolic::Expression& b) {
  return Expression(std::make_shared<const NotExpression>(
      Expression(std::make_shared<GteExpression>(a, b))));
}

symbolic::Expression IfInRange(const symbolic::Expression& index,
                               const symbolic::Expression& a,
                               const symbolic::Expression& b,
                               const symbolic::Expression& then,
                               const symbolic::Expression& ifnot) {
  const symbolic::Expression gtea(
      std::make_shared<symbolic::GteExpression>(index, a));
  const symbolic::Expression ltb = LtExpression(index, b);
  const symbolic::Expression gtea_and_ltb(
      std::make_shared<symbolic::AndExpression>(gtea, ltb));
  return symbolic::Expression(
      std::make_shared<symbolic::IfExpression>(gtea_and_ltb, then, ifnot));
}

Expression KroneckerDelta(const Expression &a, const Expression &b) {
  return a == b;
}

}  // namespace symbolic

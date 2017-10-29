#include "expression.h"
#include "numeric_value.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <experimental/optional>
#include <iostream>
#include <utility>

namespace symbolic {

Expression CreateExpression(std::string expression) {
  auto isspace = [](unsigned char const c) { return std::isspace(c); };
  expression.erase(
      std::remove_if(expression.begin(), expression.end(), isspace),
      expression.end());
  for (size_t i = 0; i < expression.size(); ++i) {
    if (expression[i] == '+') {
      auto lhs = CreateExpression(expression.substr(0, i));
      auto rhs = CreateExpression(
          expression.substr(i + 1, expression.size() - (i + 1)));
      return lhs + rhs;
    }
  }
  for (size_t i = 0; i < expression.size(); ++i) {
    if (expression[i] == '*') {
      auto lhs = CreateExpression(expression.substr(0, i));
      auto rhs = CreateExpression(
          expression.substr(i + 1, expression.size() - (i + 1)));
      return lhs * rhs;
    }
  }

  double value = 0;
  char suffix = 0;
  if (sscanf(expression.c_str(), "%lf%c", &value, &suffix) == 0) {
    // Try to sscanf an unbound variable NumericValue.
    std::string variable(expression.size(), ' ');
    if (sscanf(expression.c_str(), "%s", &variable[0]) == 0) {
      std::cerr << "Could not parse token as variable name: " << expression
                << std::endl;
      std::exit(1);
    }
    return Expression(std::make_unique<NumericValue>(variable));
  }

  // Special case for imaginary values.
  if (suffix == 'i') {
    return Expression(std::make_unique<NumericValue>(0, value));
  }

  return Expression(std::make_unique<NumericValue>(value));
}

// Expression Implementation.

Expression::Expression(std::unique_ptr<ExpressionNode>&& root)
    : expression_root_(std::move(root)) {}

Expression::Expression(const Expression& other)
    : expression_root_(other.expression_root_->Clone()) {}

Expression::Expression(Expression&& rhs)
    : expression_root_(std::move(rhs.expression_root_)) {}

Expression Expression::operator+(const Expression& rhs) const {
  auto lhscopy = expression_root_->Clone();
  auto rhscopy = rhs.expression_root_->Clone();
  return Expression(std::make_unique<AdditionExpression>(
      std::initializer_list<const ExpressionNode*>(
          {lhscopy.release(), rhscopy.release()})));
}

Expression Expression::operator*(const Expression& rhs) const {
  auto lhscopy = expression_root_->Clone();
  auto rhscopy = rhs.expression_root_->Clone();
  return Expression(std::make_unique<MultiplicationExpression>(
      std::initializer_list<const ExpressionNode*>(
          {lhscopy.release(), rhscopy.release()})));
}

Expression& Expression::operator=(const Expression& rhs) {
  expression_root_ = rhs.expression_root_->Clone();
  return *this;
}

// Variables which need to be resolved in order to evaluate the expression.
std::set<std::string> Expression::variables() const {
  return expression_root_->variables();
}

void Expression::Bind(const std::string& name, NumericValue value) {
  expression_root_ = expression_root_->Bind({{name, value}});
}

std::experimental::optional<NumericValue> Expression::Evaluate() const {
  return expression_root_->TryEvaluate();
}

std::unique_ptr<ExpressionNode> Expression::Release() {
  return std::move(expression_root_);
}

void Expression::Reset(std::unique_ptr<ExpressionNode> root) {
  expression_root_ = std::move(root);
}

std::string Expression::to_string() const {
  return expression_root_->to_string();
}

// CompoundExpression impl.

std::set<std::string> CompoundExpression::variables() const {
  std::set<std::string> variables;
  for (const auto& expression : children_) {
    std::set<std::string> child_variables = expression->variables();
    variables.insert(child_variables.begin(), child_variables.end());
  }
  return variables;
}

void CompoundExpression::add(std::unique_ptr<ExpressionNode> child) {
  children_.push_back(std::move(child));
}

std::experimental::optional<NumericValue> CompoundExpression::TryEvaluate()
    const {
  NumericValue result = identity();
  for (const auto& expression : children_) {
    const auto val_or_fail = expression->TryEvaluate();
    if (!val_or_fail) {
      return std::experimental::nullopt;
    }
    result = reduce(result, *val_or_fail);
  }
  return std::move(result);
}

std::string CompoundExpression::to_string() const {
  std::string result = "";
  for (size_t i = 0; i < children_.size(); ++i) {
    result += children_[i]->to_string();
    if (i != children_.size() - 1) {
      result += " " + operator_to_string() + " ";
    }
  }
  return result;
}

// AdditionExpression Implementation.

NumericValue AdditionExpression::reduce(const NumericValue& a,
                                        const NumericValue& b) const {
  NumericValue result;
  result.real() = a.real() + b.real();
  result.imag() = a.imag() + b.imag();
  return result;
}

std::unique_ptr<ExpressionNode> AdditionExpression::Bind(
    std::unordered_map<std::string, NumericValue> env) const {
  std::unique_ptr<AdditionExpression> b =
      std::make_unique<AdditionExpression>();
  for (const auto& expression : children_) {
    b->add(std::move(expression->Bind(env)));
  }
  return std::move(b);
}

std::unique_ptr<ExpressionNode> AdditionExpression::Derive(
    const std::string& x) const {
  std::unique_ptr<AdditionExpression> derivative_expression =
      std::make_unique<AdditionExpression>();
  for (const auto& expression : children_) {
    auto derivative = expression->Derive(x);
    // Try to evaluate the derivative. If it results in 0, discard the
    // expression. This is so that if the derivative is taken often, terms which
    // are now equal to zero don't accumulate memory.
    auto evaluation = derivative->TryEvaluate();
    if (evaluation && evaluation->real() == evaluation->imag() == 0) {
      continue;
    }
    derivative_expression->add(std::move(derivative));
  }
  return std::move(derivative_expression);
}

NumericValue AdditionExpression::identity() const { return NumericValue(0); }

// MultiplicationExpression Implementation.

NumericValue MultiplicationExpression::reduce(const NumericValue& a,
                                              const NumericValue& b) const {
  NumericValue result;
  result.real() = a.real() * b.real() - a.imag() * b.imag();
  result.imag() = a.real() * b.imag() + b.real() * a.imag();
  return result;
}

std::unique_ptr<ExpressionNode> MultiplicationExpression::Bind(
    std::unordered_map<std::string, NumericValue> env) const {
  std::unique_ptr<MultiplicationExpression> b =
      std::make_unique<MultiplicationExpression>();
  for (const auto& expression : children_) {
    b->add(std::move(expression->Bind(env)));
  }
  return std::move(b);
}

std::unique_ptr<ExpressionNode> MultiplicationExpression::Derive(
    const std::string& x) const {
  if (children_.size() < 2) {
    return children_[0]->Derive(x);
  }

  // Split A * B * C * ... into A * (B * C * ...) and use the product rule
  // recursively.

  std::vector<std::unique_ptr<ExpressionNode>> tail;
  for (size_t i = 1; i < children_.size(); ++i) {
    tail.emplace_back(children_[0]->Clone());
  }

  std::unique_ptr<const ExpressionNode> A = children_[0]->Clone();
  std::unique_ptr<const ExpressionNode> B =
      std::make_unique<MultiplicationExpression>(tail);

  std::unique_ptr<const ExpressionNode> dA = A->Derive(x);
  std::unique_ptr<const ExpressionNode> dB = B->Derive(x);  // Recursive call.

  std::unique_ptr<const ExpressionNode> A_dB =
      std::make_unique<MultiplicationExpression>(A, dB);
  std::unique_ptr<const ExpressionNode> B_dA =
      std::make_unique<MultiplicationExpression>(B, dA);

  return std::make_unique<AdditionExpression>(A_dB, B_dA);
}

NumericValue MultiplicationExpression::identity() const {
  return NumericValue(1);
}

// DivisionExpression Implementation.

std::set<std::string> DivisionExpression::variables() const {
  std::set<std::string> variables;
  std::set<std::string> numerator_variables = numerator_->variables();
  std::set<std::string> denominator_variables = denominator_->variables();

  variables.insert(numerator_variables.begin(), numerator_variables.end());
  variables.insert(denominator_variables.begin(), denominator_variables.end());
  return variables;
}

std::unique_ptr<ExpressionNode> DivisionExpression::Bind(
    std::unordered_map<std::string, NumericValue> env) const {
  std::unique_ptr<DivisionExpression> result =
      std::make_unique<DivisionExpression>();
  result->set_numerator(numerator_->Bind(env));
  result->set_denominator(denominator_->Bind(env));
  return std::move(result);
}

std::experimental::optional<NumericValue> DivisionExpression::TryEvaluate()
    const {
  NumericValue result;

  std::experimental::optional<NumericValue> numerator_result =
      numerator_->TryEvaluate();

  std::experimental::optional<NumericValue> denominator_result =
      denominator_->TryEvaluate();

  if (!(denominator_result && numerator_result)) {
    return std::experimental::nullopt;
  }

  NumericValue a = *numerator_result;
  NumericValue b = *denominator_result;

  // Fail if denominator is zero.
  if (pow(b.real(), 2) + pow(b.imag(), 2) == 0) {
    return std::experimental::nullopt;
  }

  result.real() = (a.real() * b.real() + a.imag() * b.imag()) /
                  (pow(b.real(), 2) + pow(b.imag(), 2));
  result.imag() = (a.imag() * b.real() - a.real() * b.imag()) /
                  (pow(b.real(), 2) + pow(b.imag(), 2));
  return result;
}

// Returns the symbolic partial derivative of this expression.
std::unique_ptr<ExpressionNode> DivisionExpression::Derive(
    const std::string& x) const {
  // Apply the quotient rule.
  const std::unique_ptr<const ExpressionNode>& g = numerator_;
  const std::unique_ptr<const ExpressionNode>& h = denominator_;

  std::unique_ptr<const ExpressionNode> dg = numerator_->Derive(x);
  std::unique_ptr<const ExpressionNode> dh = denominator_->Derive(x);

  std::unique_ptr<const ExpressionNode> dg_h =
      std::make_unique<MultiplicationExpression>(dg, h);
  std::unique_ptr<const ExpressionNode> g_dh =
      std::make_unique<MultiplicationExpression>(g, dh);

  std::unique_ptr<const ExpressionNode> neg =
      std::make_unique<NumericValue>(-1);
  std::unique_ptr<const ExpressionNode> neg_g_dh =
      std::make_unique<MultiplicationExpression>(neg, g_dh);

  std::unique_ptr<const ExpressionNode> h_squared =
      std::make_unique<MultiplicationExpression>(h, h);

  std::unique_ptr<const ExpressionNode> new_numerator =
      std::make_unique<AdditionExpression>(dg_h, neg_g_dh);

  return std::make_unique<DivisionExpression>(new_numerator, h_squared);
}

std::string DivisionExpression::to_string() const {
  std::string result = "(";
  result += numerator_->to_string();
  result += ") / (";
  result += denominator_->to_string();
  result += ")";
  return result;
}

}  // namespace symbolic

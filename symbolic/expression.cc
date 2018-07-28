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
    return Expression(std::make_shared<NumericValue>(variable));
  }

  // Special case for imaginary values.
  if (suffix == 'i') {
    return Expression(std::make_shared<NumericValue>(0, value));
  }

  return Expression(std::make_shared<NumericValue>(value));
}

// Expression Implementation.

Expression::Expression(std::shared_ptr<const ExpressionNode> root)
    : expression_root_(root) {}

Expression::Expression(const Expression& other)
    : expression_root_(other.expression_root_) {}

Expression::Expression(Expression&& rhs)
    : expression_root_(std::move(rhs.expression_root_)) {}

Expression::Expression(const NumericValue& rhs)
    : expression_root_(std::make_shared<NumericValue>(rhs)) {}

Expression::Expression(Number a)
    : expression_root_(std::make_shared<NumericValue>(a)) {}

Expression Expression::operator+(const Expression& rhs) const {
  return Expression(std::make_shared<AdditionExpression>(expression_root_,
                                                         rhs.expression_root_));
}

Expression Expression::operator-(const Expression& rhs) const {
  std::shared_ptr<const symbolic::ExpressionNode> neg =
      std::make_shared<symbolic::NumericValue>(-1);
  std::shared_ptr<const symbolic::ExpressionNode> rhs_neg =
      std::make_shared<MultiplicationExpression>(neg, rhs.expression_root_);
  return Expression(
      std::make_shared<AdditionExpression>(expression_root_, rhs_neg));
}

Expression Expression::operator*(const Expression& rhs) const {
  return Expression(std::static_pointer_cast<const ExpressionNode>(
      std::make_shared<const MultiplicationExpression>(expression_root_,
                                                       rhs.expression_root_)));
}

Expression Expression::operator/(const Expression& rhs) const {
  return Expression(std::make_shared<DivisionExpression>(expression_root_,
                                                         rhs.expression_root_));
}

Expression Expression::Log(NumericValue base, const Expression& exp) {
    return std::static_pointer_cast<const ExpressionNode>(std::make_shared<const LogExpression>(base, exp));
}

Expression Expression::Exp(NumericValue base, const Expression& exp) {
  return std::static_pointer_cast<const ExpressionNode>(
      std::make_shared<const ExponentExpression>(base, exp));
}

Expression& Expression::operator=(const Expression& rhs) {
  expression_root_ = rhs.expression_root_->Clone();
  return *this;
}

Expression& Expression::operator=(Expression&& rhs) {
  expression_root_ = std::move(rhs.expression_root_);
  return *this;
}

Expression& Expression::operator+=(const Expression& rhs) {
  return *this = *this + rhs;
}

// Variables which need to be resolved in order to evaluate the expression.
std::set<std::string> Expression::variables() const {
  return expression_root_->variables();
}

Expression Expression::Bind(const std::string& name, NumericValue value) const {
  return Expression(expression_root_->Bind({{name, value}}));
}

Expression Expression::Bind(const Environment& env) const {
  return Expression(expression_root_->Bind(env));
}

Expression Expression::Derive(const std::string& x) const {
  // Optimization to reduce memory consumption. If f(x) does not depend on x,
  // df(x)/dx = 0.
  std::set<std::string> unbound_vars = variables();
  if (unbound_vars.find(x) == unbound_vars.end()) {
    return Expression(0);
  }

  return Expression(expression_root_->Derive(x));
}

std::experimental::optional<NumericValue> Expression::Evaluate() const {
  return expression_root_->TryEvaluate();
}

void Expression::Reset(std::shared_ptr<const ExpressionNode> root) {
  expression_root_ = root;
}

std::string Expression::to_string() const {
  return expression_root_->to_string();
}

// IfExpression impl.

std::set<std::string> IfExpression::variables() const {
  std::set<std::string> variables;
  std::set<std::string> a_vars = a_.variables();
  std::set<std::string> b_vars = b_.variables();
  std::set<std::string> cond_vars = conditional_.variables();

  variables.insert(a_vars.begin(), a_vars.end());
  variables.insert(b_vars.begin(), b_vars.end());
  variables.insert(cond_vars.begin(), cond_vars.end());
  return variables;
}

std::shared_ptr<const ExpressionNode> IfExpression::Bind(
    const Environment& env) const {
  return std::make_shared<IfExpression>(conditional_.Bind(env), a_.Bind(env),
                                        b_.Bind(env));
}

std::experimental::optional<NumericValue> IfExpression::TryEvaluate() const {
  std::experimental::optional<NumericValue> conditional_result =
      conditional_.Evaluate();

  // Evaluate this as a truthy or falsey value. But since it's a floating point
  // number, do comparison accounting for floating point error.
  if (!conditional_result) {
    return std::experimental::nullopt;
  }

  bool truthy =
      abs(conditional_result->real()) > std::numeric_limits<Number>::epsilon();

  if (truthy) {
    return a_.Evaluate();
  } else {
    return b_.Evaluate();
  }
}

std::shared_ptr<const ExpressionNode> IfExpression::Derive(
    const std::string& x) const {
  return std::make_shared<IfExpression>(conditional_, a_.Derive(x),
                                        b_.Derive(x));
}

std::string IfExpression::to_string() const {
  return "((" + conditional_.to_string() + ") ? (" + a_.to_string() + ") : (" +
         b_.to_string() + "))";
}

// CompoundExpression impl.

std::set<std::string> CompoundExpression::variables() const {
  std::set<std::string> variables;
  std::set<std::string> head_variables = head_.variables();
  variables.insert(head_variables.begin(), head_variables.end());
  if (!is_end_) {
    std::set<std::string> tail_variables = tail_.variables();
    variables.insert(tail_variables.begin(), tail_variables.end());
  }
  return variables;
}

std::experimental::optional<NumericValue> CompoundExpression::TryEvaluate()
    const {
  auto head_or_fail = head_.Evaluate();
  auto tail_or_fail = tail_.Evaluate();
  if (!head_or_fail || !tail_or_fail) {
    return std::experimental::nullopt;
  }
  return reduce(*head_or_fail, *tail_or_fail);
}

std::string CompoundExpression::to_string() const {
  std::string result = "(";
  result += head_.to_string();
  result += ")";
  if (!is_end_) {
    result += operator_to_string();
    result += "(";
    result += tail_.to_string();
    result += ")";
  }
  return result;
}

// AdditionExpression Implementation.

NumericValue AdditionExpression::reduce(const NumericValue& a,
                                        const NumericValue& b) const {
  NumericValue result(a.real() + b.real(), a.imag() + b.imag());
  return result;
}

std::shared_ptr<const ExpressionNode> AdditionExpression::Bind(
    const Environment& env) const {
  return std::static_pointer_cast<ExpressionNode>(
      std::make_shared<AdditionExpression>(head_.Bind(env), tail_.Bind(env)));
}

std::shared_ptr<const ExpressionNode> AdditionExpression::Derive(
    const std::string& x) const {
  return std::static_pointer_cast<ExpressionNode>(
      std::make_shared<AdditionExpression>(head_.Derive(x), tail_.Derive(x)));
}

// MultiplicationExpression Implementation.

NumericValue MultiplicationExpression::reduce(const NumericValue& a,
                                              const NumericValue& b) const {
  NumericValue result(a.real() * b.real() - a.imag() * b.imag(),
                      a.real() * b.imag() + b.real() * a.imag());
  return result;
}

std::shared_ptr<const ExpressionNode> MultiplicationExpression::Bind(
    const Environment& env) const {
  return std::static_pointer_cast<ExpressionNode>(
      std::make_shared<MultiplicationExpression>(head_.Bind(env),
                                                 tail_.Bind(env)));
}

std::shared_ptr<const ExpressionNode> MultiplicationExpression::Derive(
    const std::string& x) const {
  if (is_end_) {
    return head_.Derive(x).GetPointer();
  }

  Expression dhead = head_.Derive(x);
  Expression dtail = tail_.Derive(x);  // Recursive call.

  Expression head_dtail = head_ * dtail;
  Expression tail_dhead = tail_ * dhead;

  return (head_dtail + tail_dhead).GetPointer();
}

// And Implementation.

NumericValue AndExpression::reduce(const NumericValue& a,
                                   const NumericValue& b) const {
  NumericValue result(ToNumber(ToBool(a.real()) && ToBool(b.real())), 0);
  return result;
}

std::shared_ptr<const ExpressionNode> AndExpression::Bind(
    const Environment& env) const {
  return std::make_shared<AndExpression>(head_.Bind(env), tail_.Bind(env));
}

std::shared_ptr<const ExpressionNode> AndExpression::Derive(
    const std::string& x) const {
  return nullptr;
}

// >= Expression

std::set<std::string> GteExpression::variables() const {
  std::set<std::string> variables;
  std::set<std::string> a_vars = a_.variables();
  std::set<std::string> b_vars = b_.variables();

  variables.insert(a_vars.begin(), a_vars.end());
  variables.insert(b_vars.begin(), b_vars.end());
  return variables;
}

std::shared_ptr<const ExpressionNode> GteExpression::Bind(
    const Environment& env) const {
  return std::make_shared<GteExpression>(a_.Bind(env), b_.Bind(env));
}

std::experimental::optional<NumericValue> GteExpression::TryEvaluate() const {
  std::experimental::optional<NumericValue> a_result = a_.Evaluate();

  std::experimental::optional<NumericValue> b_result = b_.Evaluate();

  if (!(a_result && b_result)) {
    return std::experimental::nullopt;
  }

  NumericValue a = *a_result;
  NumericValue b = *b_result;

  return NumericValue((a.real() >= b.real()) ? 1.0 : 0.0);
}

// Not defined for >=.
std::shared_ptr<const ExpressionNode> GteExpression::Derive(
    const std::string& x) const {
  return nullptr;
}

std::string GteExpression::to_string() const {
  std::string result = "(";
  result += a_.to_string();
  result += ") >= (";
  result += b_.to_string();
  result += ")";
  return result;
}

// DivisionExpression Implementation.

std::set<std::string> DivisionExpression::variables() const {
  std::set<std::string> variables;
  std::set<std::string> numerator_variables = numerator_.variables();
  std::set<std::string> denominator_variables = denominator_.variables();

  variables.insert(numerator_variables.begin(), numerator_variables.end());
  variables.insert(denominator_variables.begin(), denominator_variables.end());
  return variables;
}

std::shared_ptr<const ExpressionNode> DivisionExpression::Bind(
    const Environment& env) const {
  return std::make_shared<DivisionExpression>(numerator_.Bind(env),
                                              denominator_.Bind(env));
}

std::experimental::optional<NumericValue> DivisionExpression::TryEvaluate()
    const {
  std::experimental::optional<NumericValue> numerator_result =
      numerator_.Evaluate();

  std::experimental::optional<NumericValue> denominator_result =
      denominator_.Evaluate();

  if (!(denominator_result && numerator_result)) {
    return std::experimental::nullopt;
  }

  NumericValue a = *numerator_result;
  NumericValue b = *denominator_result;

  // Fail if denominator is zero.
  if (pow(b.real(), 2) + pow(b.imag(), 2) == 0) {
    return std::experimental::nullopt;
  }

  Number real = (a.real() * b.real() + a.imag() * b.imag()) /
                (pow(b.real(), 2) + pow(b.imag(), 2));
  Number imag = (a.imag() * b.real() - a.real() * b.imag()) /
                (pow(b.real(), 2) + pow(b.imag(), 2));
  return NumericValue(real, imag);
}

// Returns the symbolic partial derivative of this expression.
std::shared_ptr<const ExpressionNode> DivisionExpression::Derive(
    const std::string& x) const {
  // Apply the quotient rule.
  const Expression& g = numerator_;
  const Expression& h = denominator_;

  Expression dg = numerator_.Derive(x);
  Expression dh = denominator_.Derive(x);

  Expression dg_h = dg * h;
  Expression g_dh = g * dh;

  Expression neg_g_dh = g_dh * -1;

  Expression h_squared = h * h;

  Expression new_numerator = dg_h + neg_g_dh;

  return std::make_shared<DivisionExpression>(new_numerator, h_squared);
}

std::string DivisionExpression::to_string() const {
  std::string result = "(";
  result += numerator_.to_string();
  result += ") / (";
  result += denominator_.to_string();
  result += ")";
  return result;
}

// ExponentExpression Impl.

std::shared_ptr<const ExpressionNode> ExponentExpression::Bind(
    const Environment& env) const {
  return std::make_shared<ExponentExpression>(b_, child_.Bind(env));
}

std::experimental::optional<NumericValue> ExponentExpression::TryEvaluate()
    const {
  std::experimental::optional<NumericValue> child_result = child_.Evaluate();
  if (!child_result) {
    return std::experimental::nullopt;
  }

  // Variable names reflect the formula here:
  // http://mathworld.wolfram.com/ComplexExponentiation.html
  Number a = b_.real();
  Number b = b_.imag();

  Number c = child_result->real();
  Number d = child_result->imag();

  Number phase = atan(b / a);
  Number common = pow(a * a + b * b, c / 2) * exp(-d * phase);
  Number real = common * cos(c * phase + 0.5 * d * log(a * a + b * b));
  Number imag = common * sin(c * phase + 0.5 * d * log(a * a + b * b));

  return NumericValue(real, imag);
}

std::shared_ptr<const ExpressionNode> ExponentExpression::Derive(
    const std::string& x) const {
  Number norm = sqrt(b_.real() * b_.real() + b_.imag() * b_.imag());
  Number phase = atan(b_.imag() / b_.real());

  Expression multiplier(NumericValue(log(norm), phase));

  Expression derivative = multiplier * Expression(Clone());

  // Chain rule.
  return std::make_shared<const MultiplicationExpression>(derivative,
                                                          child_.Derive(x));
}

std::string ExponentExpression::to_string() const {
  return "pow(" + b_.to_string() + ", " + child_.to_string() + ")";
}

std::shared_ptr<const ExpressionNode> ExponentExpression::Clone() const {
  return std::make_shared<const ExponentExpression>(b_, child_);
}

// LogExpression Impl.

std::shared_ptr<const ExpressionNode> LogExpression::Bind(
    const Environment& env) const {
  return std::make_shared<LogExpression>(b_, child_.Bind(env));
}

std::experimental::optional<NumericValue> LogExpression::TryEvaluate() const {
  std::experimental::optional<NumericValue> child_result = child_.Evaluate();
  if (!child_result) {
    return std::experimental::nullopt;
  }

  if ((child_result->imag() != 0) || (b_.imag() != 0)) {
    std::cerr << "Warning: Tried evaluating a LogExpression() on a complex "
                 "expression. LogExpression is only implemented for real "
                 "numbers."
              << std::endl;
    return std::experimental::nullopt;
  }

  // https://oregonstate.edu/instruct/mth251/cq/Stage6/Lesson/logDeriv.html
  Number base = b_.real();
  Number exp = child_result->real();

  return NumericValue(log(exp) / log(base));
}

std::shared_ptr<const ExpressionNode> LogExpression::Derive(
    const std::string& x) const {
  Expression derivative =
      symbolic::CreateExpression("1") / (child_ * log(b_.real()));

  Expression child_derivative = child_.Derive(x);

  // Chain rule.
  return std::make_shared<const MultiplicationExpression>(derivative,
                                                          child_derivative);
}

std::string LogExpression::to_string() const {
  return "log(" + child_.to_string() + ") / log(" + b_.to_string() + ")";
}

std::shared_ptr<const ExpressionNode> LogExpression::Clone() const {
  return std::make_shared<const LogExpression>(b_, child_);
}

}  // namespace symbolic

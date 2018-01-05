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

Expression::Expression(std::unique_ptr<const ExpressionNode>&& root)
    : expression_root_(std::move(root)) {}

Expression::Expression(const Expression& other)
    : expression_root_(other.expression_root_->Clone()) {}

Expression::Expression(Expression&& rhs)
    : expression_root_(std::move(rhs.expression_root_)) {}

Expression::Expression(const NumericValue& rhs)
    : expression_root_(std::make_unique<NumericValue>(rhs)) {}

Expression::Expression(Number a)
    : expression_root_(std::make_unique<NumericValue>(a)) {}

Expression Expression::operator+(const Expression& rhs) const {
  return Expression(std::make_unique<AdditionExpression>(expression_root_,
                                                         rhs.expression_root_));
}

Expression Expression::operator-(const Expression& rhs) const {
  std::unique_ptr<const symbolic::ExpressionNode> neg = std::make_unique<symbolic::NumericValue>(-1);
  std::unique_ptr<const symbolic::ExpressionNode> rhs_neg =
      std::make_unique<MultiplicationExpression>(neg, rhs.expression_root_);
  return Expression(
      std::make_unique<AdditionExpression>(expression_root_, rhs_neg));
}

Expression Expression::operator*(const Expression& rhs) const {
  auto lhsresult = expression_root_->TryEvaluate();
  auto rhsresult = rhs.expression_root_->TryEvaluate();
  if (lhsresult && rhsresult && ((lhsresult->real() == 0) || (rhsresult->real() == 0))) {
    std::cerr << "optimizing out mul with zero: " << std::endl;
    return Expression(0);
  }
  return Expression(std::make_unique<MultiplicationExpression>(
      expression_root_, rhs.expression_root_));
}

Expression Expression::operator/(const Expression& rhs) const {
  auto numresult = expression_root_->TryEvaluate();
  if (numresult && (numresult->real() == 0)) {
    std::cerr << "optimizing out 0/X." << std::endl;
    return Expression(0);
  }
  return Expression(std::make_unique<DivisionExpression>(expression_root_,
                                                         rhs.expression_root_));
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
  return Expression(expression_root_->Derive(x));
}

std::experimental::optional<NumericValue> Expression::Evaluate() const {
  return expression_root_->TryEvaluate();
}

std::unique_ptr<const ExpressionNode> Expression::Release() {
  return std::move(expression_root_);
}

void Expression::Reset(std::unique_ptr<const ExpressionNode> root) {
  expression_root_ = std::move(root);
}

std::string Expression::to_string() const {
  return expression_root_->to_string();
}

// IfExpression impl.

std::set<std::string> IfExpression::variables() const {
  std::set<std::string> variables;
  std::set<std::string> a_vars = a_->variables();
  std::set<std::string> b_vars = b_->variables();
  std::set<std::string> cond_vars = conditional_->variables();

  variables.insert(a_vars.begin(), a_vars.end());
  variables.insert(b_vars.begin(), b_vars.end());
  variables.insert(cond_vars.begin(), cond_vars.end());
  return variables;
}

std::unique_ptr<const ExpressionNode> IfExpression::Bind(const Environment& env) const {
  return std::make_unique<IfExpression>(conditional_->Bind(env), a_->Bind(env),
                                        b_->Bind(env));
}

std::experimental::optional<NumericValue> IfExpression::TryEvaluate() const {
  std::experimental::optional<NumericValue> conditional_result =
      conditional_->TryEvaluate();

  // Evaluate this as a truthy or falsey value. But since it's a floating point
  // number, do comparison accounting for floating point error.
  if (!conditional_result) {
    return std::experimental::nullopt;
  }

  bool truthy = abs(conditional_result->real()) > std::numeric_limits<Number>::epsilon();

  if (truthy) {
    return a_->TryEvaluate();
  } else {
    return b_->TryEvaluate();
  }
}

std::unique_ptr<ExpressionNode> IfExpression::Derive(const std::string& x) const {
  return std::make_unique<IfExpression>(conditional_, a_->Derive(x), b_->Derive(x));
}

std::string IfExpression::to_string() const {
  return "((" + conditional_->to_string() + ") ? (" + a_->to_string() + ") : (" +
         b_->to_string() + "))";
}

// CompoundExpression impl.

std::set<std::string> CompoundExpression::variables() const {
  std::set<std::string> variables;
  std::set<std::string> head_variables = head_->variables();
  variables.insert(head_variables.begin(), head_variables.end());
  if (tail_) {
    std::set<std::string> tail_variables = tail_->variables();
    variables.insert(tail_variables.begin(), tail_variables.end());
  }
  return variables;
}

std::experimental::optional<NumericValue> CompoundExpression::TryEvaluate()
    const {
  auto head_or_fail = head_->TryEvaluate();
  auto tail_or_fail = tail_->TryEvaluate();
  if (!head_or_fail || !tail_or_fail) {
    return std::experimental::nullopt;
  }
  return reduce(*head_or_fail, *tail_or_fail);
}

std::string CompoundExpression::to_string() const {
  std::string result = "(";
  result += head_->to_string();
  result += ")";
  if (tail_) {
    result += operator_to_string();
    result += "(";
    result += tail_->to_string();
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

std::unique_ptr<const ExpressionNode> AdditionExpression::Bind(
    const Environment& env) const {
  return std::make_unique<AdditionExpression>(head_->Bind(env),
                                              tail_->Bind(env));
}

std::unique_ptr<ExpressionNode> AdditionExpression::Derive(
    const std::string& x) const {
  return std::make_unique<AdditionExpression>(head_->Derive(x),
                                              tail_->Derive(x));
}

// MultiplicationExpression Implementation.

NumericValue MultiplicationExpression::reduce(const NumericValue& a,
                                              const NumericValue& b) const {
  NumericValue result(a.real() * b.real() - a.imag() * b.imag(),
                      a.real() * b.imag() + b.real() * a.imag());
  return result;
}

std::unique_ptr<const ExpressionNode> MultiplicationExpression::Bind(
    const Environment& env) const {
  return std::make_unique<MultiplicationExpression>(head_->Bind(env),
                                                    tail_->Bind(env));
}

std::unique_ptr<ExpressionNode> MultiplicationExpression::Derive(
    const std::string& x) const {
  if (!tail_) {
    return head_->Derive(x);
  }

  std::unique_ptr<const ExpressionNode> dhead = head_->Derive(x);
  std::unique_ptr<const ExpressionNode> dtail =
      tail_->Derive(x);  // Recursive call.

  std::unique_ptr<const ExpressionNode> head_dtail =
      std::make_unique<MultiplicationExpression>(head_, dtail);
  std::unique_ptr<const ExpressionNode> tail_dhead =
      std::make_unique<MultiplicationExpression>(tail_, dhead);

  return std::make_unique<AdditionExpression>(head_dtail, tail_dhead);
}

// And Implementation.

NumericValue AndExpression::reduce(const NumericValue& a,
                                   const NumericValue& b) const {
  NumericValue result(ToNumber(ToBool(a.real()) && ToBool(b.real())), 0);
  return result;
}

std::unique_ptr<const ExpressionNode> AndExpression::Bind(
    const Environment& env) const {
  return std::make_unique<AndExpression>(head_->Bind(env), tail_->Bind(env));
}

std::unique_ptr<ExpressionNode> AndExpression::Derive(
    const std::string& x) const {
  return nullptr;
}

// >= Expression

std::set<std::string> GteExpression::variables() const {
  std::set<std::string> variables;
  std::set<std::string> a_vars = a_->variables();
  std::set<std::string> b_vars = b_->variables();

  variables.insert(a_vars.begin(), a_vars.end());
  variables.insert(b_vars.begin(), b_vars.end());
  return variables;
}

std::unique_ptr<const ExpressionNode> GteExpression::Bind(
    const Environment& env) const {
  return std::make_unique<GteExpression>(a_->Bind(env), b_->Bind(env));
}

std::experimental::optional<NumericValue> GteExpression::TryEvaluate()
    const {
  std::experimental::optional<NumericValue> a_result =
      a_->TryEvaluate();

  std::experimental::optional<NumericValue> b_result =
      b_->TryEvaluate();

  if (!(a_result && b_result)) {
    return std::experimental::nullopt;
  }

  NumericValue a = *a_result;
  NumericValue b = *b_result;

  return NumericValue((a.real() >= b.real()) ? 1.0 : 0.0);
}

// Not defined for >=.
std::unique_ptr<ExpressionNode> GteExpression::Derive(
    const std::string& x) const {
  return nullptr;
}

std::string GteExpression::to_string() const {
  std::string result = "(";
  result += a_->to_string();
  result += ") >= (";
  result += b_->to_string();
  result += ")";
  return result;
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

std::unique_ptr<const ExpressionNode> DivisionExpression::Bind(
    const Environment& env) const {
  std::unique_ptr<DivisionExpression> result =
      std::make_unique<DivisionExpression>();
  result->set_numerator(numerator_->Bind(env));
  result->set_denominator(denominator_->Bind(env));
  return std::move(result);
}

std::experimental::optional<NumericValue> DivisionExpression::TryEvaluate()
    const {
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

  Number real = (a.real() * b.real() + a.imag() * b.imag()) /
                (pow(b.real(), 2) + pow(b.imag(), 2));
  Number imag = (a.imag() * b.real() - a.real() * b.imag()) /
                (pow(b.real(), 2) + pow(b.imag(), 2));
  return NumericValue(real, imag);
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

// ExponentExpression Impl.

std::set<std::string> ExponentExpression::variables() const {
  return child_->variables();
}

std::unique_ptr<const ExpressionNode> ExponentExpression::Bind(
    const Environment& env) const {
  return std::make_unique<ExponentExpression>(b_, child_->Bind(env));
}

std::experimental::optional<NumericValue> ExponentExpression::TryEvaluate()
    const {
  std::experimental::optional<NumericValue> child_result =
      child_->TryEvaluate();
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

std::unique_ptr<ExpressionNode> ExponentExpression::Derive(
    const std::string& x) const {
  Number norm = sqrt(b_.real() * b_.real() + b_.imag() * b_.imag());
  Number phase = atan(b_.imag() / b_.real());

  const std::unique_ptr<const ExpressionNode> multiplier =
      std::make_unique<NumericValue>(log(norm), phase);

  std::unique_ptr<const ExpressionNode> derivative =
      std::make_unique<MultiplicationExpression>(Clone(), multiplier);

  std::unique_ptr<const ExpressionNode> chain_rule = child_->Derive(x);

  return std::make_unique<MultiplicationExpression>(derivative, chain_rule);
}

std::string ExponentExpression::to_string() const {
  return "pow(" + b_.to_string() + ", " + child_->to_string() + ")";
}

std::unique_ptr<const ExpressionNode> ExponentExpression::Clone() const {
  return std::make_unique<ExponentExpression>(b_, child_);
}

}  // namespace symbolic

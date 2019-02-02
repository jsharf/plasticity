#include "expression.h"
#include "numeric_value.h"

#include <algorithm>
#include <cctype>
#include <cmath>
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

Expression::Expression(std::unique_ptr<const ExpressionNode>&& root)
    : expression_root_(std::shared_ptr<const ExpressionNode>(root.release())) {}

Expression::Expression(std::shared_ptr<const ExpressionNode> root)
    : expression_root_(root) {}

Expression::Expression(const Expression& other)
    : expression_root_(other.expression_root_) {}

Expression::Expression(Expression&& rhs)
    : expression_root_(std::move(rhs.expression_root_)) {}

Expression::Expression(const NumericValue& rhs)
    : expression_root_(std::make_shared<NumericValue>(rhs)) {}

Expression::Expression(const Integer& rhs)
    : expression_root_(std::make_shared<Integer>(rhs)) {}

Expression::Expression(double a)
    : expression_root_(std::make_shared<NumericValue>(a)) {}

Expression::Expression(int a)
    : expression_root_(std::make_shared<Integer>(a)) {}

Expression::Expression(unsigned long a)
    : expression_root_(std::make_shared<Integer>(a)) {}

Expression Expression::CreateInteger(const std::string& name) {
  return Expression(std::make_shared<Integer>(name));
}

Expression Expression::CreateNumericValue(const std::string& name) {
  return Expression(std::make_shared<NumericValue>(name));
}

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

Expression Expression::operator%(const Expression& rhs) const {
  return Expression(std::make_shared<ModulusExpression>(expression_root_,
                                                        rhs.expression_root_));
}

Expression& Expression::operator=(const Expression& rhs) {
  expression_root_ = std::shared_ptr<const ExpressionNode>(
      rhs.expression_root_->Clone().release());
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

Expression Expression::Bind(const std::string& name,
                            const NumericValue& value) const {
  std::unordered_map<std::string, std::unique_ptr<NumericValue>> env_pointers;
  env_pointers.emplace(name, value.CloneValue());
  return Expression(expression_root_->Bind(env_pointers));
}

Expression Expression::Bind(const Environment& env) const {
  std::unordered_map<std::string, std::unique_ptr<NumericValue>> env_pointers;
  for (auto env_entry : env) {
    env_pointers.emplace(env_entry.first,
                         std::make_unique<NumericValue>(env_entry.second));
  }
  return Expression(expression_root_->Bind(env_pointers));
}

Expression Expression::Bind(
    const std::unordered_map<std::string, std::unique_ptr<NumericValue>>& env)
    const {
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

std::unique_ptr<NumericValue> Expression::Evaluate() const {
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
    const std::unordered_map<std::string, std::unique_ptr<NumericValue>>& env)
    const {
  return std::make_shared<IfExpression>(conditional_.Bind(env), a_.Bind(env),
                                        b_.Bind(env));
}

std::unique_ptr<NumericValue> IfExpression::TryEvaluate() const {
  std::unique_ptr<NumericValue> conditional_result = conditional_.Evaluate();

  // Evaluate this as a truthy or falsey value. But since it's a floating point
  // number, do comparison accounting for floating point error.
  if (!conditional_result) {
    return nullptr;
  }

  bool truthy =
      abs(conditional_result->real()) > std::numeric_limits<double>::epsilon();

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

std::unique_ptr<NumericValue> CompoundExpression::TryEvaluate() const {
  auto head_or_fail = head_.Evaluate();
  auto tail_or_fail = tail_.Evaluate();
  if (!head_or_fail || !tail_or_fail) {
    return nullptr;
  }
  // head is cloned to preserve type information.
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

std::unique_ptr<NumericValue> AdditionExpression::reduce(
    const NumericValue& a, const NumericValue& b) const {
  // This preserves the type (integer vs double).
  std::unique_ptr<NumericValue> result = a.CloneValue();
  result->real() = a.real() + b.real();
  result->imag() = a.imag() + b.imag();
  return result;
}

std::shared_ptr<const ExpressionNode> AdditionExpression::Bind(
    const std::unordered_map<std::string, std::unique_ptr<NumericValue>>& env)
    const {
  return std::static_pointer_cast<ExpressionNode>(
      std::make_shared<AdditionExpression>(head_.Bind(env), tail_.Bind(env)));
}

std::shared_ptr<const ExpressionNode> AdditionExpression::Derive(
    const std::string& x) const {
  return std::static_pointer_cast<ExpressionNode>(
      std::make_shared<AdditionExpression>(head_.Derive(x), tail_.Derive(x)));
}

// MultiplicationExpression Implementation.

std::unique_ptr<NumericValue> MultiplicationExpression::reduce(
    const NumericValue& a, const NumericValue& b) const {
  // This preserves the type (integer vs double).
  std::unique_ptr<NumericValue> result = a.CloneValue();
  result->real() = a.real() * b.real() - a.imag() * b.imag();
  result->imag() = a.real() * b.imag() + b.real() * a.imag();
  return result;
}

std::shared_ptr<const ExpressionNode> MultiplicationExpression::Bind(
    const std::unordered_map<std::string, std::unique_ptr<NumericValue>>& env)
    const {
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

std::unique_ptr<NumericValue> AndExpression::reduce(
    const NumericValue& a, const NumericValue& b) const {
  return std::make_unique<NumericValue>(
      ToNumber(ToBool(a.real()) && ToBool(b.real())), 0);
}

std::shared_ptr<const ExpressionNode> AndExpression::Bind(
    const std::unordered_map<std::string, std::unique_ptr<NumericValue>>& env)
    const {
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
    const std::unordered_map<std::string, std::unique_ptr<NumericValue>>& env)
    const {
  return std::make_shared<GteExpression>(a_.Bind(env), b_.Bind(env));
}

std::unique_ptr<NumericValue> GteExpression::TryEvaluate() const {
  std::unique_ptr<NumericValue> a_result = a_.Evaluate();

  std::unique_ptr<NumericValue> b_result = b_.Evaluate();

  if (!(a_result && b_result)) {
    return nullptr;
  }

  NumericValue a = *a_result;
  NumericValue b = *b_result;

  return std::make_unique<Integer>((a.real() >= b.real()) ? 1.0 : 0.0);
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

// == Expression

std::set<std::string> EqExpression::variables() const {
  std::set<std::string> variables;
  std::set<std::string> a_vars = a_.variables();
  std::set<std::string> b_vars = b_.variables();

  variables.insert(a_vars.begin(), a_vars.end());
  variables.insert(b_vars.begin(), b_vars.end());
  return variables;
}

std::shared_ptr<const ExpressionNode> EqExpression::Bind(
    const std::unordered_map<std::string, std::unique_ptr<NumericValue>>& env)
    const {
  return std::make_shared<EqExpression>(a_.Bind(env), b_.Bind(env));
}

std::unique_ptr<NumericValue> EqExpression::TryEvaluate() const {
  std::unique_ptr<NumericValue> a_result = a_.Evaluate();

  std::unique_ptr<NumericValue> b_result = b_.Evaluate();

  if (!(a_result && b_result)) {
    return nullptr;
  }

  NumericValue a = *a_result;
  NumericValue b = *b_result;

  return std::make_unique<Integer>((a.real() == b.real()) ? 1.0 : 0.0);
}

// Not defined for ==.
std::shared_ptr<const ExpressionNode> EqExpression::Derive(
    const std::string& x) const {
  return nullptr;
}

std::string EqExpression::to_string() const {
  std::string result = "(";
  result += a_.to_string();
  result += ") == (";
  result += b_.to_string();
  result += ")";
  return result;
}

// Not Expression

std::set<std::string> NotExpression::variables() const {
  return child_.variables();
}

std::shared_ptr<const ExpressionNode> NotExpression::Bind(
    const std::unordered_map<std::string, std::unique_ptr<NumericValue>>& env)
    const {
  return std::make_shared<NotExpression>(child_.Bind(env));
}

std::unique_ptr<NumericValue> NotExpression::TryEvaluate() const {
  std::unique_ptr<NumericValue> child_result = child_.Evaluate();

  if (!child_result) {
    return nullptr;
  }

  bool truthy =
      abs(child_result->real()) > std::numeric_limits<double>::epsilon();

  return std::make_unique<Integer>((truthy) ? 0 : 1);
}

std::string NotExpression::to_string() const {
  return "!(" + child_.to_string() + ")";
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
    const std::unordered_map<std::string, std::unique_ptr<NumericValue>>& env)
    const {
  return std::make_shared<DivisionExpression>(numerator_.Bind(env),
                                              denominator_.Bind(env));
}

std::unique_ptr<NumericValue> DivisionExpression::TryEvaluate() const {
  std::unique_ptr<NumericValue> numerator_result = numerator_.Evaluate();

  std::unique_ptr<NumericValue> denominator_result = denominator_.Evaluate();

  if (!(denominator_result && numerator_result)) {
    return nullptr;
  }

  NumericValue a = *numerator_result;
  NumericValue b = *denominator_result;

  // Fail if denominator is zero.
  if (pow(b.real(), 2) + pow(b.imag(), 2) == 0) {
    std::cerr << "Divide by zero error in evaluation of symbol: " << to_string()
              << std::endl;
    return nullptr;
  }

  double real = (a.real() * b.real() + a.imag() * b.imag()) /
                (pow(b.real(), 2) + pow(b.imag(), 2));
  double imag = (a.imag() * b.real() - a.real() * b.imag()) /
                (pow(b.real(), 2) + pow(b.imag(), 2));
  // This is done so that the type of the operands is preserved. This allows
  // ints to stay as ints.
  numerator_result->real() = real;
  numerator_result->imag() = imag;
  return numerator_result;
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

// ModulusExpression Implementation.

std::set<std::string> ModulusExpression::variables() const {
  std::set<std::string> variables;
  std::set<std::string> a_variables = a_.variables();
  std::set<std::string> b_variables = b_.variables();

  variables.insert(a_variables.begin(), a_variables.end());
  variables.insert(b_variables.begin(), b_variables.end());
  return variables;
}

std::shared_ptr<const ExpressionNode> ModulusExpression::Bind(
    const std::unordered_map<std::string, std::unique_ptr<NumericValue>>& env)
    const {
  return std::make_shared<ModulusExpression>(a_.Bind(env), b_.Bind(env));
}

std::unique_ptr<NumericValue> ModulusExpression::TryEvaluate() const {
  std::unique_ptr<NumericValue> a_result = a_.Evaluate();

  std::unique_ptr<NumericValue> b_result = b_.Evaluate();

  if (!(b_result && a_result)) {
    return nullptr;
  }

  NumericValue a = *a_result;
  NumericValue b = *b_result;

  // Fail if either value has an imaginary component.
  if ((a.imag() != 0) || (b.imag() != 0)) {
    std::cerr << "Modulus taken of value with imaginary component!"
              << std::endl;
    return nullptr;
  }

  // Fail if b is zero.
  if (b.real() == 0) {
    return nullptr;
  }

  // This is done so that the type of the operands is preserved. This allows
  // ints to stay as ints.
  a_result->real() = static_cast<int>(a.real()) % static_cast<int>(b.real());
  a_result->imag() = 0;
  return a_result;
}

std::string ModulusExpression::to_string() const {
  std::string result = "(";
  result += a_.to_string();
  result += ") % (";
  result += b_.to_string();
  result += ")";
  return result;
}

// ExponentExpression Impl.

std::shared_ptr<const ExpressionNode> ExponentExpression::Bind(
    const std::unordered_map<std::string, std::unique_ptr<NumericValue>>& env)
    const {
  return std::make_shared<ExponentExpression>(b_, child_.Bind(env));
}

std::unique_ptr<NumericValue> ExponentExpression::TryEvaluate() const {
  std::unique_ptr<NumericValue> child_result = child_.Evaluate();
  if (!child_result) {
    return nullptr;
  }

  // Variable names reflect the formula here:
  // http://mathworld.wolfram.com/ComplexExponentiation.html
  double a = b_.real();
  double b = b_.imag();

  double c = child_result->real();
  double d = child_result->imag();

  double phase = atan(b / a);
  double common = pow(a * a + b * b, c / 2) * exp(-d * phase);
  double real = common * cos(c * phase + 0.5 * d * log(a * a + b * b));
  double imag = common * sin(c * phase + 0.5 * d * log(a * a + b * b));

  // child_result is cloned to preserve type.
  std::unique_ptr<NumericValue> result = std::move(child_result);
  result->real() = real;
  result->imag() = imag;
  return result;
}

std::shared_ptr<const ExpressionNode> ExponentExpression::Derive(
    const std::string& x) const {
  double norm = sqrt(b_.real() * b_.real() + b_.imag() * b_.imag());
  double phase = atan(b_.imag() / b_.real());

  Expression multiplier(NumericValue(log(norm), phase));

  Expression derivative = multiplier * Expression(Clone());

  // Chain rule.
  return std::make_shared<const MultiplicationExpression>(derivative,
                                                          child_.Derive(x));
}

std::string ExponentExpression::to_string() const {
  return "pow(" + b_.to_string() + ", " + child_.to_string() + ")";
}

std::unique_ptr<const ExpressionNode> ExponentExpression::Clone() const {
  return std::make_unique<const ExponentExpression>(b_, child_);
}

// LogExpression Impl.

std::shared_ptr<const ExpressionNode> LogExpression::Bind(
    const std::unordered_map<std::string, std::unique_ptr<NumericValue>>& env)
    const {
  return std::make_shared<LogExpression>(b_, child_.Bind(env));
}

std::unique_ptr<NumericValue> LogExpression::TryEvaluate() const {
  std::unique_ptr<NumericValue> child_result = child_.Evaluate();
  if (!child_result) {
    return nullptr;
  }

  if (child_result->real() <= 0) {
    std::cerr << "Error: tried evaluating log of a negative number! "
                 "LogExpression failed to TryEvaluate()."
              << std::endl;
    std::cerr << "\tExpression: " << child_.to_string() << std::endl;
    std::cerr << "\tValue: " << child_result->real() << std::endl;
    return nullptr;
  }

  if ((child_result->imag() != 0) || (b_.imag() != 0)) {
    std::cerr << "Error: Tried evaluating a LogExpression() on a complex "
                 "expression. LogExpression is only implemented for real "
                 "numbers."
              << std::endl;
    std::cerr << "\tExpression: " << child_.to_string() << std::endl;
    std::cerr << "\tValue: " << child_result->real() << std::endl;
    return nullptr;
  }

  // https://oregonstate.edu/instruct/mth251/cq/Stage6/Lesson/logDeriv.html
  double base = b_.real();
  double exp = child_result->real();

  // child_result is cloned to preserve type.
  child_result->real() = log(exp) / log(base);
  child_result->imag() = 0;
  return child_result;
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

std::unique_ptr<const ExpressionNode> LogExpression::Clone() const {
  return std::make_unique<const LogExpression>(b_, child_);
}

}  // namespace symbolic

#include "expression.h"
#include "numeric_value.h"

#include <algorithm>
#include <cctype>

namespace symbolic {

std::unique_ptr<Expression> CreateExpression(std::string expression) {  
  auto isspace = [](unsigned char const c) { return std::isspace(c); };
  expression.erase(std::remove_if(expression.begin(), expression.end(), isspace), expression.end());
  for (size_t i = 0; i < expression.size(); ++i) {
    if (expression[i] == '+') {
      auto exprs = {CreateExpression(expression.substr(0, i)).release(), CreateExpression(expression.substr(i+1, expression.size() - (i + 1))).release()};
      return std::move(std::make_unique<AdditionExpression>(exprs));
    }
  }
  for (size_t i = 0; i < expression.size(); ++i) {
    if (expression[i] == '*') {
      auto exprs = {CreateExpression(expression.substr(0, i)).release(), CreateExpression(expression.substr(i+1, expression.size() - (i + 1))).release()};
      return std::move(std::make_unique<MultiplicationExpression>(exprs));
    }
  }

  double value = 0;
  char suffix = 0;
  if (sscanf(expression.c_str(), "%lf%c", &value, &suffix) == 0) {
    // Try to sscanf an unbound variable NumericValue.
    std::string variable(expression.size(), ' ');
    if (sscanf(expression.c_str(), "%s", &variable[0]) == 0) {
      return nullptr;
    }
    return std::make_unique<NumericValue>(variable);
  }

  // Special case for imaginary values.
  if (suffix == 'i') {
    return std::move(std::make_unique<NumericValue>(0, value));
  }
  
  return std::move(std::make_unique<NumericValue>(value));
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

void CompoundExpression::add(std::unique_ptr<Expression> child) {
  children_.push_back(std::move(child));
}

std::experimental::optional<NumericValue> CompoundExpression::TryEvaluate() const {
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

std::unique_ptr<Expression> AdditionExpression::Bind(
    std::unordered_map<std::string, NumericValue> env) const {
  std::unique_ptr<AdditionExpression> b = std::make_unique<AdditionExpression>();
  for (const auto& expression : children_) {
    b->add(std::move(expression->Bind(env)));
  }
  return std::move(b);
}

NumericValue AdditionExpression::identity() const {
  return NumericValue(0);
}

// MultiplicationExpression Implementation.

NumericValue MultiplicationExpression::reduce(const NumericValue& a,
                    const NumericValue& b) const {
  NumericValue result;
  result.real() = a.real() * b.real() - a.imag() * b.imag();
  result.imag() = a.real() * b.imag() + b.real() * a.imag();
  return result;
}

std::unique_ptr<Expression> MultiplicationExpression::Bind(
    std::unordered_map<std::string, NumericValue> env) const {
  std::unique_ptr<MultiplicationExpression> b = std::make_unique<MultiplicationExpression>();
  for (const auto& expression : children_) {
    b->add(std::move(expression->Bind(env)));
  }
  return std::move(b);
}
  
NumericValue MultiplicationExpression::identity() const {
  return NumericValue(1);
}

}  // namespace symbolic

#include "math/symbolic/numeric_value.h"

namespace symbolic {

std::unique_ptr<ExpressionNode> NumericValue::Bind(
    std::unordered_map<std::string, NumericValue> env) const {
  if (!is_bound_) {
    if (env.count(name_) == 1) {
      Number a = env[name_].real();
      Number b = env[name_].imag();
      return std::move(std::make_unique<NumericValue>(a, b));
    }
  }
  return Clone();
}

std::set<std::string> NumericValue::variables() const {
  return std::set<std::string>{name_};
}

std::experimental::optional<NumericValue> NumericValue::TryEvaluate()
    const {
  if (!is_bound_) {
    return std::experimental::nullopt;
  }
  return *this;
}

// Returns the symbolic partial derivative of this expression.
std::unique_ptr<ExpressionNode> NumericValue::Derive(
    const std::string& x) const {
  if (!is_bound_ && (name_ == x)) {
    return std::make_unique<NumericValue>(1);
  }
  return std::make_unique<NumericValue>(0);
}

std::string NumericValue::to_string() const {
  if (!is_bound_) {
    return name_;
  }
  std::string result = std::to_string(a_);
  if (b_ != 0) {
    result += " + " + std::to_string(b_) + "i";
  }
  return result;
}

std::unique_ptr<ExpressionNode> NumericValue::Clone() const {
  if (is_bound_) {
    return std::move(std::make_unique<NumericValue>(a_, b_));
  } else {
    return std::move(std::make_unique<NumericValue>(name_));
  }
}

const NumericValue NumericValue::pi(3.141592653589793238);
const NumericValue NumericValue::e(2.718281828459045235);

}  // namespace symbolic

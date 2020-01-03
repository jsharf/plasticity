#include "plasticity/symbolic/integer.h"
#include <sstream>

namespace symbolic {

// Returns the symbolic partial derivative of this expression.
std::shared_ptr<const ExpressionNode> Integer::Derive(
    const std::string& x) const {
  if (!is_bound_ && (name_ == x)) {
    return std::make_shared<Integer>(1);
  }
  return std::make_shared<Integer>(0);
}

std::string Integer::to_string() const {
  if (!is_bound_) {
    return name_;
  }
  std::ostringstream result;
  result << static_cast<int>(a_);
  return result.str();
}

std::unique_ptr<const ExpressionNode> Integer::Clone() const {
  if (is_bound_) {
    return std::make_unique<Integer>(static_cast<int>(a_));
  } else {
    return std::make_unique<Integer>(name_);
  }
}

std::unique_ptr<NumericValue> Integer::CloneValue() const {
  if (is_bound_) {
    return std::make_unique<Integer>(static_cast<int>(a_));
  } else {
    return std::make_unique<Integer>(name_);
  }
}

}  // namespace symbolic

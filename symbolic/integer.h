#ifndef INTEGER_H
#define INTEGER_H

#include "symbolic/expression_node.h"
#include "symbolic/numeric_value.h"

#include <cmath>
#include <optional>
#include <iostream>
#include <set>
#include <string>
#include <unordered_map>

namespace symbolic {

class Integer : public NumericValue {
 public:
  Integer(int a) : NumericValue(a) {}
  Integer(std::string name) : NumericValue(name) {}
  Integer(const Integer& rhs) : NumericValue(rhs) {}
  Integer() : NumericValue() {}

  int get() { return static_cast<int>(NumericValue::a_); }

  std::unique_ptr<NumericValue> TryEvaluate() const override {
    auto val = NumericValue::TryEvaluate();
    if (val) {
      return std::make_unique<Integer>(static_cast<int>(val->real()));
    }
    return val;
  }

  double& real() override { 
    a_ = std::trunc(a_);
    return a_;
  }
  double& imag() override {
    b_ = std::trunc(b_);
    return b_;
  }
  double real() const override {
    return std::trunc(a_);
  }
  double imag() const override {
    return std::trunc(b_);
  }

  // Returns the symbolic partial derivative of this expression.
  std::shared_ptr<const ExpressionNode> Derive(
      const std::string& x) const override;

  std::string to_string() const override;

  std::unique_ptr<NumericValue> CloneValue() const override;

  std::unique_ptr<const ExpressionNode> Clone() const override;
};

}  // namespace symbolic

#endif /* INTEGER_H */

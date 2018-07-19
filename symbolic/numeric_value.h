#ifndef NUMERIC_VALUE_H
#define NUMERIC_VALUE_H

#include "math/symbolic/expression_node.h"

#include <experimental/optional>
#include <iostream>
#include <set>
#include <string>
#include <unordered_map>

namespace symbolic {

using Number = double;

class NumericValue : public ExpressionNode {
 public:
  explicit NumericValue(Number a) : is_bound_(true), a_(a), b_(0) {}
  NumericValue(Number a, Number b) : is_bound_(true), a_(a), b_(b) {}
  NumericValue(std::string name) : is_bound_(false), name_(name) {}
  NumericValue() : is_bound_(true), a_(0), b_(0) {}
  Number& real() { return a_; }
  Number& imag() { return b_; }
  Number real() const { return a_; }
  Number imag() const { return b_; }

  std::shared_ptr<const ExpressionNode> Bind(
      const std::unordered_map<std::string, NumericValue>& env) const override;

  std::set<std::string> variables() const override;

  std::experimental::optional<NumericValue> TryEvaluate() const override;

  // Returns the symbolic partial derivative of this expression.
  std::shared_ptr<const ExpressionNode> Derive(const std::string& x) const override;

  std::string to_string() const override;

  std::shared_ptr<const ExpressionNode> Clone() const override;

  static const NumericValue pi;
  static const NumericValue e;

 private:
  bool is_bound_;

  // For unbound variables.
  std::string name_;

  // For bound variables with actual values.
  Number a_;
  Number b_;
};

}  // namespace symbolic

#endif /* NUMERIC_VALUE_H */

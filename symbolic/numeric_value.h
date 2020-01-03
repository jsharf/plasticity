#ifndef NUMERIC_VALUE_H
#define NUMERIC_VALUE_H

#include "plasticity/symbolic/expression_node.h"

#include <experimental/optional>
#include <iostream>
#include <set>
#include <string>
#include <unordered_map>

namespace symbolic {

class NumericValue : public ExpressionNode {
 public:
  NumericValue(double a) : is_bound_(true), a_(a), b_(0) {}
  NumericValue(double a, double b) : is_bound_(true), a_(a), b_(b) {}
  NumericValue(std::string name) : is_bound_(false), name_(name) {}
  NumericValue() : is_bound_(true), a_(0), b_(0) {}
  NumericValue(const NumericValue& rhs)
      : is_bound_(rhs.is_bound_), name_(rhs.name_), a_(rhs.a_), b_(rhs.b_) {}
  virtual double& real() { return a_; }
  virtual double& imag() { return b_; }
  virtual double real() const { return a_; }
  virtual double imag() const { return b_; }

  std::shared_ptr<const ExpressionNode> Bind(
      const std::unordered_map<std::string, std::unique_ptr<NumericValue>>& env)
      const override;

  std::set<std::string> variables() const override;

  std::unique_ptr<NumericValue> TryEvaluate() const override;

  // Returns the symbolic partial derivative of this expression.
  std::shared_ptr<const ExpressionNode> Derive(
      const std::string& x) const override;

  std::string to_string() const override;

  virtual std::unique_ptr<NumericValue> CloneValue() const;

  std::unique_ptr<const ExpressionNode> Clone() const override;

  static const NumericValue pi;
  static const NumericValue e;

 protected:
  bool is_bound_;

  // For unbound variables.
  std::string name_;

  // For bound variables with actual values.
  double a_;
  double b_;
};

}  // namespace symbolic

#endif /* NUMERIC_VALUE_H */

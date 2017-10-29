#ifndef NUMERIC_VALUE_H
#define NUMERIC_VALUE_H

#include <experimental/optional>
#include <set>
#include <string>
#include <unordered_map>
#include <iostream>

#include "expression.h"

namespace symbolic {

using Number = double;

class ExpressionNode;

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
  virtual std::unique_ptr<ExpressionNode> Bind(std::unordered_map<std::string, NumericValue> env) const override {
    if (!is_bound_) {
      if (env.count(name_) == 1) {
        Number a = env[name_].real();
        Number b = env[name_].imag();
        return std::move(std::make_unique<NumericValue>(a, b));
      }
    }
    return Clone();
  }

  virtual std::set<std::string> variables() const override {
    return std::set<std::string>{name_};
  }

  virtual std::experimental::optional<NumericValue> TryEvaluate() const override {
    if (!is_bound_) {
      return std::experimental::nullopt;
    }
    return *this;
  }
  
  // Returns the symbolic partial derivative of this expression.
  std::unique_ptr<ExpressionNode> Derive(
      const std::string& x) const override {
    if (!is_bound_ && (name_ == x)) {
      return std::make_unique<NumericValue>(1);
    }
    return std::make_unique<NumericValue>(0); 
  }

  virtual std::string to_string() const override {
    if (!is_bound_) {
      return name_;
    }
    std::string result = std::to_string(a_);
    if (b_ != 0) {
      result += " + " + std::to_string(b_) + "i";
    }
    return result;
  }
 
  virtual std::unique_ptr<ExpressionNode> Clone() const override {
    if (is_bound_) {
      return std::move(std::make_unique<NumericValue>(a_, b_));
    } else {
      return std::move(std::make_unique<NumericValue>(name_));
    }
  }

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

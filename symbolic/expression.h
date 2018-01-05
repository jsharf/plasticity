#ifndef EXPRESSION_H
#define EXPRESSION_H

#include "math/symbolic/expression_node.h"
#include "math/symbolic/numeric_value.h"

#include <experimental/optional>
#include <iostream>
#include <limits>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace symbolic {

// Used to map the environment in which an expression is evaluated (binds
// variable names to their values).
//
// For instance, x^2 + y evaluated in the environment { x: "2", y: "3"} becomes
// 2^2 + 3, or 7.
using Environment = std::unordered_map<std::string, symbolic::NumericValue>;

class AdditionExpression;
class Expression;

Expression CreateExpression(std::string expression);

// Class which holds the ExpressionNode tree and provides an easy-to-use
// interface.
class Expression {
 public:
  Expression() : Expression(0) {}

  Expression(std::unique_ptr<const ExpressionNode>&& root);

  Expression(const Expression& other);

  explicit Expression(Expression&& rhs);

  Expression(const NumericValue& value);

  Expression(Number a);

  Expression operator+(const Expression& rhs) const;
  Expression operator-(const Expression& rhs) const;
  Expression operator*(const Expression& rhs) const;
  Expression operator/(const Expression& rhs) const;

  Expression& operator=(const Expression& rhs);
  Expression& operator=(Expression&& rhs);

  Expression& operator+=(const Expression& rhs);

  friend std::ostream& operator<<(std::ostream& output, const Expression& exp) {
    output << exp.to_string();
    return output;
  }

  // Variables which need to be resolved in order to evaluate the expression.
  std::set<std::string> variables() const;

  Expression Bind(const std::string& name, NumericValue value) const;

  Expression Bind(const Environment& env) const;

  Expression Derive(const std::string& x) const;

  std::experimental::optional<NumericValue> Evaluate() const;

  std::unique_ptr<const ExpressionNode> Release();

  void Reset(std::unique_ptr<const ExpressionNode> root);

  const std::unique_ptr<const ExpressionNode>& GetPointer() const {
    return expression_root_;
  }

  std::string to_string() const;

 private:
  std::unique_ptr<const ExpressionNode> expression_root_;
};

class IfExpression : public ExpressionNode {
 public:
  // if(conditional) { a } else { b }
  IfExpression(const std::unique_ptr<const ExpressionNode>& conditional,
               const std::unique_ptr<const ExpressionNode>& a,
               const std::unique_ptr<const ExpressionNode>& b)
      : conditional_(conditional->Clone()), a_(a->Clone()), b_(b->Clone()) {}

  // Variables which need to be resolved in order to evaluate the expression.
  std::set<std::string> variables() const override;

  // Bind variables to values to create an expression which can be evaluated.
  std::unique_ptr<const ExpressionNode> Bind(
      const Environment& env) const override;

  // If all variables in the expression have been bound, this produces a
  // numerical evaluation of the expression.
  std::experimental::optional<NumericValue> TryEvaluate() const override;

  // Returns the symbolic partial derivative of this expression.
  // Note: This does not do any bounds analysis and simply returns
  // IfExpression(conditional_, a_->Derive(x), b_->Derive(x)).
  std::unique_ptr<ExpressionNode> Derive(const std::string& x) const override;

  // Converts IfExpression to string form:
  // (( conditional ) ? ( a ) : ( b ))
  std::string to_string() const override;

  std::unique_ptr<const ExpressionNode> Clone() const override {
    std::unique_ptr<IfExpression> clone =
        std::make_unique<IfExpression>(conditional_, a_, b_);
    return std::move(clone);
  }

 private:
  std::unique_ptr<const ExpressionNode> conditional_;
  std::unique_ptr<const ExpressionNode> a_;
  std::unique_ptr<const ExpressionNode> b_;
};

class CompoundExpression : public ExpressionNode {
 public:
  std::set<std::string> variables() const override;

  virtual std::unique_ptr<const ExpressionNode> Bind(
      const Environment& env) const override = 0;

  std::experimental::optional<NumericValue> TryEvaluate() const override;

  std::string to_string() const override;

  virtual NumericValue reduce(const NumericValue& a,
                              const NumericValue& b) const = 0;

  virtual std::string operator_to_string() const = 0;

  virtual std::unique_ptr<const ExpressionNode> Clone() const override = 0;

 protected:
  CompoundExpression(const std::unique_ptr<const ExpressionNode>& a,
                     const std::unique_ptr<const ExpressionNode>& b)
      : head_(a->Clone()), tail_(b->Clone()) {}
  CompoundExpression(const std::unique_ptr<const ExpressionNode>& head)
      : head_(head->Clone()) {}
  CompoundExpression() {}
  std::unique_ptr<const ExpressionNode> head_;
  std::unique_ptr<const ExpressionNode> tail_;
};

class AdditionExpression : public CompoundExpression {
 public:
  AdditionExpression(const std::unique_ptr<const ExpressionNode>& a,
                     const std::unique_ptr<const ExpressionNode>& b)
      : CompoundExpression(a, b) {}
  AdditionExpression(const std::unique_ptr<const ExpressionNode>& a)
      : CompoundExpression(a) {}
  AdditionExpression() : CompoundExpression() {}
  NumericValue reduce(const NumericValue& a,
                      const NumericValue& b) const override;
  std::string operator_to_string() const override { return "+"; }

  std::unique_ptr<const ExpressionNode> Clone() const override {
    if (!tail_) {
      return std::make_unique<AdditionExpression>(head_);
    }
    return std::make_unique<AdditionExpression>(head_, tail_);
  }

  std::unique_ptr<const ExpressionNode> Bind(
      const Environment& env) const override;

  // Returns the symbolic partial derivative of this expression.
  std::unique_ptr<ExpressionNode> Derive(const std::string& x) const override;
};

class MultiplicationExpression : public CompoundExpression {
 public:
  MultiplicationExpression(const std::unique_ptr<const ExpressionNode>& a,
                           const std::unique_ptr<const ExpressionNode>& b)
      : CompoundExpression(a, b) {}

  MultiplicationExpression(const std::unique_ptr<const ExpressionNode>& a)
      : CompoundExpression(a) {}

  MultiplicationExpression() : CompoundExpression() {}

  NumericValue reduce(const NumericValue& a,
                      const NumericValue& b) const override;

  std::string operator_to_string() const override { return "*"; }

  std::unique_ptr<const ExpressionNode> Clone() const override {
    if (!tail_) {
      return std::make_unique<MultiplicationExpression>(head_);
    }
    return std::make_unique<MultiplicationExpression>(head_, tail_);
  }

  std::unique_ptr<const ExpressionNode> Bind(
      const Environment& env) const override;

  // Returns the symbolic partial derivative of this expression.
  std::unique_ptr<ExpressionNode> Derive(const std::string& x) const override;
};

// Boolean &&
class AndExpression : public CompoundExpression {
 public:
  AndExpression(const std::unique_ptr<const ExpressionNode>& a,
                const std::unique_ptr<const ExpressionNode>& b)
      : CompoundExpression(a, b) {}

  AndExpression(const std::unique_ptr<const ExpressionNode>& a)
      : CompoundExpression(a) {}

  AndExpression() : CompoundExpression() {}

  NumericValue reduce(const NumericValue& a,
                      const NumericValue& b) const override;

  std::string operator_to_string() const override { return "&&"; }

  std::unique_ptr<const ExpressionNode> Clone() const override {
    if (!tail_) {
      return std::make_unique<AndExpression>(head_);
    }
    return std::make_unique<AndExpression>(head_, tail_);
  }

  static bool ToBool(Number x) {
    return abs(x) > std::numeric_limits<Number>::epsilon();
  }

  static Number ToNumber(bool b) {
    return b ? 1.0 : 0.0;
  }

  std::unique_ptr<const ExpressionNode> Bind(
      const Environment& env) const override;

  // Returns the symbolic partial derivative of this expression.
  std::unique_ptr<ExpressionNode> Derive(const std::string& x) const override;
};

// >= Conditional expression

class GteExpression : public ExpressionNode {
 public:
  GteExpression(const std::unique_ptr<const ExpressionNode>& a,
                const std::unique_ptr<const ExpressionNode>& b)
      : a_(a->Clone()), b_(b->Clone()) {}

  // Variables which need to be resolved in order to evaluate the expression.
  std::set<std::string> variables() const override;

  // Bind variables to values to create an expression which can be evaluated.
  std::unique_ptr<const ExpressionNode> Bind(
      const Environment& env) const override;

  // If all variables in the expression have been bound, this produces a
  // numerical evaluation of the expression.
  std::experimental::optional<NumericValue> TryEvaluate() const override;

  // Not defined for GteExpression. Returns nullptr.
  std::unique_ptr<ExpressionNode> Derive(const std::string& x) const override;

  std::string to_string() const override;

  std::unique_ptr<const ExpressionNode> Clone() const override {
    std::unique_ptr<GteExpression> clone =
        std::make_unique<GteExpression>(a_, b_);
    return std::move(clone);
  }

 private:
  std::unique_ptr<const ExpressionNode> a_;
  std::unique_ptr<const ExpressionNode> b_;
};

class DivisionExpression : public ExpressionNode {
 public:
  DivisionExpression(const std::unique_ptr<const ExpressionNode>& numerator,
                     const std::unique_ptr<const ExpressionNode>& denominator)
      : numerator_(numerator->Clone()), denominator_(denominator->Clone()) {}

  DivisionExpression() {}

  void set_numerator(std::unique_ptr<const ExpressionNode>&& numerator) {
    numerator_ = std::move(numerator);
  }

  void set_denominator(std::unique_ptr<const ExpressionNode>&& denominator) {
    denominator_ = std::move(denominator);
  }

  // Variables which need to be resolved in order to evaluate the expression.
  std::set<std::string> variables() const override;

  // Bind variables to values to create an expression which can be evaluated.
  std::unique_ptr<const ExpressionNode> Bind(
      const Environment& env) const override;

  // If all variables in the expression have been bound, this produces a
  // numerical evaluation of the expression.
  std::experimental::optional<NumericValue> TryEvaluate() const override;

  // Returns the symbolic partial derivative of this expression.
  std::unique_ptr<ExpressionNode> Derive(const std::string& x) const override;

  std::string to_string() const override;

  std::unique_ptr<const ExpressionNode> Clone() const override {
    std::unique_ptr<DivisionExpression> clone =
        std::make_unique<DivisionExpression>(numerator_, denominator_);
    return std::move(clone);
  }

 private:
  std::unique_ptr<const ExpressionNode> numerator_;
  std::unique_ptr<const ExpressionNode> denominator_;
};

// b^x.
class ExponentExpression : public ExpressionNode {
 public:
  ExponentExpression(const NumericValue& b,
                     const std::unique_ptr<const ExpressionNode>& child)
      : b_(b), child_(child->Clone()) {}
  // Variables which need to be resolved in order to evaluate the expression.
  std::set<std::string> variables() const override;
  // Bind variables to values to create an expression which can be evaluated.
  std::unique_ptr<const ExpressionNode> Bind(
      const Environment& env) const override;
  // If all variables in the expression have been bound, this produces a
  // numerical evaluation of the expression.
  std::experimental::optional<NumericValue> TryEvaluate() const override;

  // Returns the symbolic partial derivative of this expression.
  std::unique_ptr<ExpressionNode> Derive(const std::string& x) const override;

  std::string to_string() const override;

  std::unique_ptr<const ExpressionNode> Clone() const override;

 private:
  NumericValue b_;
  std::unique_ptr<const ExpressionNode> child_;
};

}  // namespace symbolic

#endif /* EXPRESSION_H */

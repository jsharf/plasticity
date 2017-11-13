#ifndef EXPRESSION_H
#define EXPRESSION_H

#include "math/symbolic/expression_node.h"
#include "math/symbolic/numeric_value.h"

#include <experimental/optional>
#include <iostream>
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

  Expression() {}

  Expression(std::unique_ptr<ExpressionNode>&& root);

  Expression(const Expression& other);

  explicit Expression(Expression&& rhs);

  Expression(const NumericValue& value);

  Expression(Number a);

  Expression operator+(const Expression& rhs) const;

  Expression operator*(const Expression& rhs) const;

  Expression operator/(const Expression& rhs) const;

  Expression& operator=(const Expression& rhs);

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

  std::unique_ptr<ExpressionNode> Release();

  void Reset(std::unique_ptr<ExpressionNode> root);

  std::string to_string() const;

 private:
  std::unique_ptr<ExpressionNode> expression_root_;
};

class CompoundExpression : public ExpressionNode {
 public:
  std::set<std::string> variables() const override;

  virtual std::unique_ptr<ExpressionNode> Bind(
      const Environment& env) const = 0;

  void add(std::unique_ptr<ExpressionNode> child);

  std::experimental::optional<NumericValue> TryEvaluate() const override;

  std::string to_string() const override;

  virtual NumericValue reduce(const NumericValue& a,
                              const NumericValue& b) const = 0;

  virtual std::string operator_to_string() const = 0;

  virtual std::unique_ptr<ExpressionNode> Clone() const = 0;

  virtual NumericValue identity() const = 0;

 protected:
  CompoundExpression(std::initializer_list<const ExpressionNode*> arguments) {
    for (const ExpressionNode* exp : arguments) {
      children_.emplace_back(exp->Clone());
    }
  }
  CompoundExpression(
      const std::vector<std::unique_ptr<ExpressionNode>>& children) {
    for (const std::unique_ptr<ExpressionNode>& child : children) {
      children_.emplace_back(child->Clone());
    }
  }
  CompoundExpression(const std::unique_ptr<const ExpressionNode>& a,
                     const std::unique_ptr<const ExpressionNode>& b) {
    children_.emplace_back(a->Clone());
    children_.emplace_back(b->Clone());
  }
  CompoundExpression() {}
  std::vector<std::unique_ptr<ExpressionNode>> children_;
};

class AdditionExpression : public CompoundExpression {
 public:
  AdditionExpression(std::initializer_list<const ExpressionNode*> arguments)
      : CompoundExpression(arguments) {}
  AdditionExpression(
      const std::vector<std::unique_ptr<ExpressionNode>>& children)
      : CompoundExpression(children) {}
  AdditionExpression(const std::unique_ptr<const ExpressionNode>& a,
                     const std::unique_ptr<const ExpressionNode>& b)
      : CompoundExpression(a, b) {}
  AdditionExpression() : CompoundExpression() {}
  NumericValue reduce(const NumericValue& a,
                      const NumericValue& b) const override;
  std::string operator_to_string() const override { return "+"; }

  std::unique_ptr<ExpressionNode> Clone() const override {
    std::unique_ptr<AdditionExpression> clone =
        std::make_unique<AdditionExpression>();
    for (auto& child : children_) {
      clone->add(child->Clone());
    }
    return std::move(clone);
  }

  std::unique_ptr<ExpressionNode> Bind(const Environment& env) const override;

  // Returns the symbolic partial derivative of this expression.
  std::unique_ptr<ExpressionNode> Derive(const std::string& x) const override;

  NumericValue identity() const override;
};

class MultiplicationExpression : public CompoundExpression {
 public:
  MultiplicationExpression(
      std::initializer_list<const ExpressionNode*> arguments)
      : CompoundExpression(arguments) {}
  MultiplicationExpression(
      const std::vector<std::unique_ptr<ExpressionNode>>& children)
      : CompoundExpression(children) {}
  MultiplicationExpression(const std::unique_ptr<const ExpressionNode>& a,
                           const std::unique_ptr<const ExpressionNode>& b)
      : CompoundExpression(a, b) {}

  MultiplicationExpression() : CompoundExpression() {}

  NumericValue reduce(const NumericValue& a,
                      const NumericValue& b) const override;

  std::string operator_to_string() const override { return "*"; }

  std::unique_ptr<ExpressionNode> Clone() const override {
    std::unique_ptr<MultiplicationExpression> clone =
        std::make_unique<MultiplicationExpression>();
    for (auto& child : children_) {
      clone->add(child->Clone());
    }
    return std::move(clone);
  }

  std::unique_ptr<ExpressionNode> Bind(const Environment& env) const override;

  // Returns the symbolic partial derivative of this expression.
  std::unique_ptr<ExpressionNode> Derive(const std::string& x) const override;

  NumericValue identity() const override;
};

class DivisionExpression : public ExpressionNode {
 public:
  DivisionExpression(const std::unique_ptr<const ExpressionNode>& numerator,
                     const std::unique_ptr<const ExpressionNode>& denominator)
      : numerator_(numerator->Clone()), denominator_(denominator->Clone()) {}

  DivisionExpression() {}

  void set_numerator(std::unique_ptr<ExpressionNode>&& numerator) {
    numerator_ = std::move(numerator);
  }

  void set_denominator(std::unique_ptr<ExpressionNode>&& denominator) {
    denominator_ = std::move(denominator);
  }

  // Variables which need to be resolved in order to evaluate the expression.
  std::set<std::string> variables() const override;

  // Bind variables to values to create an expression which can be evaluated.
  std::unique_ptr<ExpressionNode> Bind(const Environment& env) const override;

  // If all variables in the expression have been bound, this produces a
  // numerical evaluation of the expression.
  std::experimental::optional<NumericValue> TryEvaluate() const override;

  // Returns the symbolic partial derivative of this expression.
  std::unique_ptr<ExpressionNode> Derive(const std::string& x) const override;

  std::string to_string() const override;

  std::unique_ptr<ExpressionNode> Clone() const override {
    std::unique_ptr<DivisionExpression> clone =
        std::make_unique<DivisionExpression>();
    clone->set_numerator(numerator_->Clone());
    clone->set_denominator(denominator_->Clone());
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
                     std::unique_ptr<const ExpressionNode> child)
      : b_(b), child_(child->Clone()) {}
  // Variables which need to be resolved in order to evaluate the expression.
  std::set<std::string> variables() const override;
  // Bind variables to values to create an expression which can be evaluated.
  std::unique_ptr<ExpressionNode> Bind(const Environment& env) const override;
  // If all variables in the expression have been bound, this produces a
  // numerical evaluation of the expression.
  std::experimental::optional<NumericValue> TryEvaluate() const override;

  // Returns the symbolic partial derivative of this expression.
  std::unique_ptr<ExpressionNode> Derive(const std::string& x) const override;

  std::string to_string() const override;

  std::unique_ptr<ExpressionNode> Clone() const override;

 private:
  NumericValue b_;
  std::unique_ptr<const ExpressionNode> child_;
};

}  // namespace symbolic

#endif /* EXPRESSION_H */

#ifndef EXPRESSION_H
#define EXPRESSION_H

#include <experimental/optional>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace symbolic {

class AdditionExpression;
class Expression;
class ExpressionNode;
class NumericValue;

Expression CreateExpression(std::string expression);

// Class which holds the ExpressionNode tree and provides an easy-to-use
// interface.
class Expression {
 public:
  Expression() {}

  Expression(std::unique_ptr<ExpressionNode>&& root);

  Expression(const Expression& other);

  explicit Expression(Expression&& rhs);

  Expression operator+(const Expression& rhs) const;

  Expression operator*(const Expression& rhs) const;

  Expression& operator=(const Expression& rhs);

  // Variables which need to be resolved in order to evaluate the expression.
  std::set<std::string> variables() const;

  void Bind(const std::string& name, NumericValue value);

  std::experimental::optional<NumericValue> Evaluate() const;

  std::string to_string() const;

 private:
  std::unique_ptr<ExpressionNode> expression_root_;
};

// Abstract class which defines the expression interface. Interface is limited
// to prevent accidental copies (inefficiencies). Not optimized for ease of use.
class ExpressionNode {
 public:
  // Variables which need to be resolved in order to evaluate the expression.
  virtual std::set<std::string> variables() const = 0;
  // Bind variables to values to create an expression which can be evaluated.
  virtual std::unique_ptr<ExpressionNode> Bind(
      std::unordered_map<std::string, NumericValue>) const = 0;
  // If all variables in the expression have been bound, this produces a
  // numerical evaluation of the expression.
  virtual std::experimental::optional<NumericValue> TryEvaluate() const = 0;

  virtual std::string to_string() const = 0;

  virtual std::unique_ptr<ExpressionNode> Clone() const = 0;
};

class CompoundExpression : public ExpressionNode {
 public:
  std::set<std::string> variables() const override;

  virtual std::unique_ptr<ExpressionNode> Bind(
      std::unordered_map<std::string, NumericValue> env) const = 0;

  void add(std::unique_ptr<ExpressionNode> child);

  std::experimental::optional<NumericValue> TryEvaluate() const override;

  std::string to_string() const override;

  virtual NumericValue reduce(const NumericValue& a,
                              const NumericValue& b) const = 0;

  virtual std::string operator_to_string() const = 0;

  virtual std::unique_ptr<ExpressionNode> Clone() const = 0;

  virtual NumericValue identity() const = 0;

 protected:
  CompoundExpression(std::initializer_list<ExpressionNode*> arguments) {
    for (ExpressionNode* exp : arguments) {
      children_.emplace_back(exp);
    }
  }
  CompoundExpression() {}
  std::vector<std::unique_ptr<ExpressionNode>> children_;
};

class AdditionExpression : public CompoundExpression {
 public:
  AdditionExpression(std::initializer_list<ExpressionNode*> arguments)
      : CompoundExpression(arguments) {}
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

  std::unique_ptr<ExpressionNode> Bind(
      std::unordered_map<std::string, NumericValue> env) const override;

  NumericValue identity() const override;
};

class MultiplicationExpression : public CompoundExpression {
 public:
  MultiplicationExpression(std::initializer_list<ExpressionNode*> arguments)
      : CompoundExpression(arguments) {}
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

  std::unique_ptr<ExpressionNode> Bind(
      std::unordered_map<std::string, NumericValue> env) const override;

  NumericValue identity() const override;
};

}  // namespace symbolic

#endif /* EXPRESSION_H */

#ifndef EXPRESSION_H
#define EXPRESSION_H

#include <experimental/optional>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace symbolic {

class NumericValue;
class Expression;

std::unique_ptr<Expression> CreateExpression(std::string expression);

// Abstract class which defines the expression interface.
class Expression {
 public:
  // Variables which need to be resolved in order to evaluate the expression.
  virtual std::set<std::string> variables() const = 0;
  // Bind variables to values to create an expression which can be evaluated.
  virtual std::unique_ptr<Expression> Bind(
      std::unordered_map<std::string, NumericValue>) const = 0;
  // If all variables in the expression have been bound, this produces a
  // numerical evaluation of the expression.
  virtual std::experimental::optional<NumericValue> TryEvaluate() const = 0;

  virtual std::string to_string() const = 0;

  virtual std::unique_ptr<Expression> Clone() const = 0;
};

class CompoundExpression : public Expression {
 public:
  std::set<std::string> variables() const override;

  virtual std::unique_ptr<Expression> Bind(
      std::unordered_map<std::string, NumericValue> env) const = 0;

  void add(std::unique_ptr<Expression> child);

  std::experimental::optional<NumericValue> TryEvaluate() const override;

  std::string to_string() const override;

  virtual NumericValue reduce(const NumericValue& a,
                              const NumericValue& b) const = 0;

  virtual std::string operator_to_string() const = 0;

  virtual std::unique_ptr<Expression> Clone() const = 0;

  virtual NumericValue identity() const = 0;
protected:
  CompoundExpression(std::initializer_list<Expression*> arguments) {
    for (Expression* exp : arguments) {
      children_.emplace_back(exp);
    }
  }
  CompoundExpression() {}
  std::vector<std::unique_ptr<Expression>> children_;
};

class AdditionExpression : public CompoundExpression {
 public:
  AdditionExpression(std::initializer_list<Expression*> arguments)
      : CompoundExpression(arguments) {}
  AdditionExpression() : CompoundExpression() {}
  NumericValue reduce(const NumericValue& a,
                      const NumericValue& b) const override;
  std::string operator_to_string() const override { return "+"; }

  std::unique_ptr<Expression> Clone() const override {
    std::unique_ptr<AdditionExpression> clone =
        std::make_unique<AdditionExpression>();
    for (auto& child : children_) {
      clone->add(child->Clone());
    }
    return std::move(clone);
  }

  std::unique_ptr<Expression> Bind(
      std::unordered_map<std::string, NumericValue> env) const override;
  
  NumericValue identity() const override;
};

class MultiplicationExpression : public CompoundExpression {
 public:
  MultiplicationExpression(std::initializer_list<Expression*> arguments)
      : CompoundExpression(arguments) {}
  MultiplicationExpression() : CompoundExpression() {}
  NumericValue reduce(const NumericValue& a,
                      const NumericValue& b) const override;
  std::string operator_to_string() const override { return "*"; }

  std::unique_ptr<Expression> Clone() const override {
    std::unique_ptr<MultiplicationExpression> clone =
        std::make_unique<MultiplicationExpression>();
    for (auto& child : children_) {
      clone->add(child->Clone());
    }
    return std::move(clone);
  }

  std::unique_ptr<Expression> Bind(
      std::unordered_map<std::string, NumericValue> env) const override;
  
  NumericValue identity() const override;
};

}  // namespace symbolic

#endif /* EXPRESSION_H */

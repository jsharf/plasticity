#ifndef EXPRESSION_NODE_H
#define EXPRESSION_NODE_H

#include <optional>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>

namespace symbolic {

class NumericValue;

// Abstract class which defines the expression interface. Interface is limited
// to prevent accidental copies (inefficiencies). Not optimized for ease of use.
class ExpressionNode {
 public:
  // Variables which need to be resolved in order to evaluate the expression.
  virtual std::set<std::string> variables() const = 0;
  // Bind variables to values to create an expression which can be evaluated.
  virtual std::shared_ptr<const ExpressionNode> Bind(
      const std::unordered_map<std::string, std::unique_ptr<NumericValue>>&) const = 0;
  // If all variables in the expression have been bound, this produces a
  // numerical evaluation of the expression.
  virtual std::unique_ptr<NumericValue> TryEvaluate() const = 0;

  // Returns the symbolic partial derivative of this expression.
  virtual std::shared_ptr<const ExpressionNode> Derive(
      const std::string& x) const = 0;

  virtual std::string to_string() const = 0;

  virtual std::unique_ptr<const ExpressionNode> Clone() const = 0;

  virtual ~ExpressionNode() {}
};

}  // namespace symbolic

#endif /* EXPRESSION_NODE_H */

#include <cstdlib>

#include <iostream>
#include <set>
#include <memory>
#include <experimental/optional>

#include "numeric_value.h"
#include "expression.h"

using symbolic::Expression;
using symbolic::NumericValue;

int main() {
  Expression equation = symbolic::CreateExpression("a*x + b + 0.5i");

  std::cout << "Equation: \n" << equation.to_string() << std::endl;

  std::cout << "Binding..." << std::endl;

  equation.Bind("a", NumericValue(1));
  equation.Bind("b", NumericValue(1));
  equation.Bind("x", NumericValue(1));
  
  std::cout << "Fixed Equation: \n" << equation.to_string() << std::endl;

  std::cout << "Evaluating..." << std::endl;

  std::experimental::optional<NumericValue> result = equation.Evaluate();
  if (result) {
    std::cout << "Result of ax + b, a = 1, b = 0.5, x = pi: " << result->to_string() << std::endl;
  }
}

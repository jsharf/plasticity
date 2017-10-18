#include <iostream>
#include <set>
#include <memory>
#include <experimental/optional>

#include "numeric_value.h"
#include "expression.h"

using symbolic::Expression;
using symbolic::NumericValue;

int main() {
  auto equation = symbolic::CreateExpression("a*x + b + 0.5i");

  std::cout << "Equation: \n" << equation->to_string() << std::endl;

  std::cout << "Binding..." << std::endl;

  std::unique_ptr<Expression> fixed_equation = equation->Bind(
      {{"a", NumericValue(1)},
       {"b", NumericValue(0.5)},
       {"x", NumericValue(3.141592)},
       });
  
  std::cout << "Fixed Equation: \n" << fixed_equation->to_string() << std::endl;

  std::cout << "Evaluating..." << std::endl;

  std::experimental::optional<NumericValue> result = fixed_equation->TryEvaluate();
  if (result) {
    std::cout << "Result of ax + b, a = 1, b = 0.5, x = pi: " << result->to_string() << std::endl;
  }
}

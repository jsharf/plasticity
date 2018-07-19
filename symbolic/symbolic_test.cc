#include <cstdlib>

#include <experimental/optional>
#include <iostream>
#include <memory>
#include <set>

#include "math/symbolic/expression.h"
#include "math/symbolic/numeric_value.h"
#include "math/symbolic/symbolic_util.h"

using symbolic::Expression;
using symbolic::GteExpression;
using symbolic::IfExpression;
using symbolic::NumericValue;

int main() {
  Expression equation = symbolic::CreateExpression("a*x + b + 0.5i");

  std::cout << "Equation: \n" << equation.to_string() << std::endl;

  std::cout << "Binding..." << std::endl;

  equation = equation.Bind("a", NumericValue(1));
  std::cout << "b" << std::endl;
  equation = equation.Bind("b", NumericValue(1));
  std::cout << "x" << std::endl;
  equation = equation.Bind("x", NumericValue(1));

  std::cout << "Fixed Equation: \n" << equation.to_string() << std::endl;

  std::cout << "Evaluating..." << std::endl;

  std::experimental::optional<NumericValue> result = equation.Evaluate();
  if (result) {
    std::cout << "Result of ax + b + 0.5i, a = 1, b = 1, x = 1: "
              << result->to_string() << std::endl;
  }

  Expression eq1 = symbolic::CreateExpression("x * x");
  Expression eq2 = symbolic::CreateExpression("25");
  Expression pieceweise_fn = Expression(std::make_shared<IfExpression>(
      Expression(std::make_shared<GteExpression>(
          Expression(5), Expression(NumericValue("x")))),
      eq1, eq2));
  Expression pieceweise_deriv = pieceweise_fn.Derive("x");

  std::cout << "f(x) = " << pieceweise_fn.to_string() << std::endl;
  std::cout << "Printing f(x) and f'(x) for x = [0, 30]" << std::endl;
  std::cout << "x" << '\t' << "f(x)" << '\t' << "f'(x)" << std::endl;
  for (size_t x = 0; x < 30; ++x) {
    std::cout
        << x << '\t'
        << pieceweise_fn.Bind("x", NumericValue(x)).Evaluate()->to_string()
        << '\t'
        << pieceweise_deriv.Bind("x", NumericValue(x)).Evaluate()->to_string()
        << std::endl;
  }

  Expression max_test = symbolic::Max(
      {symbolic::CreateExpression("3"), symbolic::CreateExpression("2"),
       symbolic::CreateExpression("1"), symbolic::CreateExpression("5")});

  std::cout << "max(3, 2, 1, 5) = " << std::endl
            << max_test.to_string() << std::endl;

  std::cout << "Eval = " << std::endl
            << max_test.Evaluate()->real() << std::endl;

  std::cout << "DONE" << std::endl;
}

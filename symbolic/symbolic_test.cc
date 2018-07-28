#define CATCH_CONFIG_MAIN
#include "math/third_party/catch.h"

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

TEST_CASE("Simple expression output is validated", "[symbolic]") {
  Expression equation = symbolic::CreateExpression("a*x + b + 0.5i");

  std::cout << "Equation: \n" << equation.to_string() << std::endl;

  std::cout << "Binding..." << std::endl;

  equation = equation.Bind("a", NumericValue(1));
  std::cout << "b" << std::endl;
  equation = equation.Bind("b", NumericValue(1));
  std::cout << "x" << std::endl;
  equation = equation.Bind("x", NumericValue(1));

  std::experimental::optional<NumericValue> result = equation.Evaluate();
  REQUIRE(result);
  REQUIRE(result->real() == 2);
  REQUIRE(result->imag() == Approx(0.5));
}

TEST_CASE("Pieceweise function is evaluated", "[symbolic]") {
  Expression eq1 = symbolic::CreateExpression("x * x");
  Expression eq2 = symbolic::CreateExpression("25");
  Expression pieceweise_fn = Expression(std::make_shared<IfExpression>(
      Expression(std::make_shared<GteExpression>(
          Expression(5), Expression(NumericValue("x")))),
      eq1, eq2));
  Expression pieceweise_deriv = pieceweise_fn.Derive("x");

  for (size_t x = 0; x < 30; ++x) {
    if (x <= 5) {
      REQUIRE(pieceweise_fn.Bind("x", x).Evaluate()->real() == Approx(x * x));
      REQUIRE(pieceweise_deriv.Bind("x", x).Evaluate()->real() ==
              Approx(2 * x));
    } else {
      REQUIRE(pieceweise_fn.Bind("x", x).Evaluate()->real() == 25);
      REQUIRE(pieceweise_deriv.Bind("x", x).Evaluate()->real() == 0);
    }
  }
}

TEST_CASE("Test max expression", "[symbolic]") {
  Expression max_test = symbolic::Max(
      {symbolic::CreateExpression("3"), symbolic::CreateExpression("2"),
       symbolic::CreateExpression("1"), symbolic::CreateExpression("5")});

  REQUIRE(max_test.Evaluate()->real() == 5);
}

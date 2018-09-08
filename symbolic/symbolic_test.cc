#define CATCH_CONFIG_MAIN
#include "math/third_party/catch.h"

#include <cstdlib>

#include <iostream>
#include <memory>
#include <set>

#include "math/symbolic/expression.h"
#include "math/symbolic/integer.h"
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

  std::unique_ptr<NumericValue> result = equation.Evaluate();
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

TEST_CASE("integer arithmetic validation", "[symbolic]") {
  size_t a = 10;
  size_t b = 7;
  size_t c = 4;

  SECTION("VERIFY SIMPLE DIVISION") {
    Expression expr = symbolic::Integer(a);
    Expression quotient = expr / b;
    std::unique_ptr<NumericValue> result = quotient.Evaluate();
    REQUIRE(result);
    REQUIRE(result->real() == 1);
    quotient = expr / c;
    result = quotient.Evaluate();
    REQUIRE(result);
    REQUIRE(result->real() == 2);
  }
}

TEST_CASE("3D Array index flatten & unflatten", "[symbolic]") {
  size_t width = 10;
  size_t height = 5;
  size_t depth = 3;

  SECTION("VERIFY FLATTEN") {
    symbolic::Expression flattened_index = symbolic::Flatten3d(
        width, height, depth, symbolic::Expression("row"),
        symbolic::Expression("col"), symbolic::Expression("plane"));
    flattened_index =
        flattened_index.Bind({{"row", 2}, {"col", 3}, {"plane", 2}});
    auto result = flattened_index.Evaluate();
    REQUIRE(result);
    REQUIRE(result->real() == 123);
  }

  SECTION("VERIFY UNFLATTEN") {
    symbolic::Integer flattened_index_value(120);

    symbolic::Expression flattened_index = symbolic::Expression("index");

    symbolic::Expression row =
        symbolic::Unflatten3dRow(width, height, depth, flattened_index);
    row = row.Bind("index", flattened_index_value);
    auto row_result = row.Evaluate();
    REQUIRE(row_result);
    REQUIRE(row_result->real() == 2);

    symbolic::Expression col =
        symbolic::Unflatten3dCol(width, height, depth, flattened_index);
    col = col.Bind("index", flattened_index_value);
    auto col_result = col.Evaluate();
    REQUIRE(col_result);
    REQUIRE(col_result->real() == 0);

    symbolic::Expression plane =
        symbolic::Unflatten3dPlane(width, height, depth, flattened_index);
    plane = plane.Bind("index", flattened_index_value);
    auto plane_result = plane.Evaluate();
    REQUIRE(plane_result);
    REQUIRE(plane_result->real() == 2);
  }
}

TEST_CASE("2D Array index flatten & unflatten", "[symbolic]") {
  size_t width = 10;
  size_t height = 5;

  SECTION("VERIFY FLATTEN") {
    symbolic::Expression flattened_index =
        symbolic::Flatten2d(width, height, symbolic::Expression("row"),
                            symbolic::Expression("col"));
    flattened_index =
        flattened_index.Bind({{"row", 2}, {"col", 3}});
    auto result = flattened_index.Evaluate();
    REQUIRE(result);
    REQUIRE(result->real() == 23);
  }

  SECTION("VERIFY UNFLATTEN") {
    symbolic::Integer flattened_index_value(34);

    symbolic::Expression flattened_index = symbolic::Expression("index");

    symbolic::Expression row =
        symbolic::Unflatten2dRow(width, height, flattened_index);
    row = row.Bind("index", flattened_index_value);
    auto row_result = row.Evaluate();
    REQUIRE(row_result);
    REQUIRE(row_result->real() == 3);

    symbolic::Expression col =
        symbolic::Unflatten2dCol(width, height, flattened_index);
    col = col.Bind("index", flattened_index_value);
    auto col_result = col.Evaluate();
    REQUIRE(col_result);
    REQUIRE(col_result->real() == 4);
  }
}

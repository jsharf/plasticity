#include <iostream>
#include <memory>
#include <string>

#include "symbolic/expression.h"
#include "symbolic/numeric_value.h"
#include "geometry/dynamic_matrix.h"

using std::string;
using symbolic::Expression;
using symbolic::NumericValue;
using Number=double;

int main() {
  Matrix<double> test = {
      {3, -0.1, -0.2}, {0.1, 7, -0.3}, {0.3, -0.2, 10},
  };

  // <lower, upper>
  std::pair<Matrix<double>, Matrix<double>> LU = test.LUDecomp();

  string lower = LU.first.to_string();
  string upper = LU.second.to_string();
  Matrix<double> b = {{7.85}, {-19.3}, {71.4}};
  Matrix<double> x = test.LUSolve(b);
  std::cout << "A:" << std::endl << test.to_string() << std::endl;
  std::cout << "Lower:" << std::endl << lower << std::endl;
  std::cout << "Upper:" << std::endl << upper << std::endl;
  std::cout << "b = " << std::endl << b.to_string() << std::endl;
  std::cout << "Ax = b, x = " << std::endl << x.to_string() << std::endl;
  std::cout << "INV(A) = " << std::endl
            << test.Invert().to_string() << std::endl;

  Expression zero = symbolic::CreateExpression("0");

  Matrix<Expression> scale(3, 3, zero);

  // Makes a symbolic scaling matrix of the form:
  //  [xscale 0 0]
  //  [0 yscale 0]
  //  [0 0 zscale]
  scale.at(0, 0) = symbolic::CreateExpression("xscale");
  scale.at(1, 1) = symbolic::CreateExpression("yscale");
  scale.at(2, 2) = symbolic::CreateExpression("zscale");

  std::function<Number(const Expression&)> evaluator =
      [](const Expression& exp) -> Number {
    Expression clone = exp;
    clone = clone.Bind("xscale", NumericValue(0.5));
    clone = clone.Bind("yscale", NumericValue(1));
    clone = clone.Bind("zscale", NumericValue(11.2));
    auto value = clone.Evaluate();
    if (!value) {
      std::cerr << "Failed to evaluate expression." << std::endl;
      std::exit(1);
    }
    return value->real();
  };

  Matrix<Number> myscalematrix = scale.Map(evaluator);

  std::cout << myscalematrix.to_string() << std::endl;
}

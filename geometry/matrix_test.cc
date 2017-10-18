#include <iostream>
#include <string>
#include <unordered_map>
#include <memory>

#include "matrix.h"
#include "math/symbolic/expression.h"
#include "math/symbolic/numeric_value.h"

using std::string;
using symbolic::Expression;
using symbolic::NumericValue;
using symbolic::Number;

int main() {
  using Mat3 = Matrix<3, 3, double>;
  using ColVec = Matrix<3, 1, double>;
  Mat3 test = {
    { 3, -0.1, -0.2 },
    { 0.1, 7, -0.3},
    { 0.3, -0.2, 10 },
  };

  // <lower, upper>
  std::pair<Mat3, Mat3> LU = test.LUDecomp();
  
  string lower = LU.first.to_string();
  string upper = LU.second.to_string();
  ColVec b = { {7.85}, {-19.3}, {71.4} };
  ColVec x = test.LUSolve(b);
  std::cout << "A:" << std::endl << test.to_string() << std::endl;
  std::cout << "Lower:" << std::endl << lower << std::endl;
  std::cout << "Upper:" << std::endl << upper << std::endl;
  std::cout << "b = " << std::endl << b.to_string() << std::endl;
  std::cout << "Ax = b, x = " << std::endl << x.to_string() << std::endl;
  std::cout << "INV(A) = " << std::endl << test.Invert().to_string() << std::endl;

  Matrix<3, 3, std::unique_ptr<Expression>> symbols;

  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      symbols.at(i, j) = symbolic::CreateExpression("0");
    }
  }

  // Makes a symbolic scaling matrix of the form:
  //  [xscale 0 0]
  //  [0 yscale 0]
  //  [0 0 zscale]
  symbols.at(0, 0) = symbolic::CreateExpression("xscale");
  symbols.at(1, 1) = symbolic::CreateExpression("yscale");
  symbols.at(2, 2) = symbolic::CreateExpression("zscale");
  
  std::function<Number(const std::unique_ptr<Expression>&)> evaluator = +[](const std::unique_ptr<Expression>& exp) -> Number {
    std::unordered_map<string, NumericValue> env = {
      {"xscale", NumericValue(0.5) },
      {"yscale", NumericValue(2) },
      {"zscale", NumericValue(3.14) },
    };
    auto fixed_exp = exp->Bind(env);
    return fixed_exp->TryEvaluate()->real();
  };
  
  Matrix<3, 3, Number> myscalematrix = symbols.map(evaluator);

  std::cout << myscalematrix.to_string() << std::endl;

}

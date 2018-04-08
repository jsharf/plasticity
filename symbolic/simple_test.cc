#include <cstdlib>

#include <iostream>
#include <set>
#include <memory>
#include <experimental/optional>

#include "math/symbolic/numeric_value.h"
#include "math/symbolic/expression.h"
#include "math/symbolic/symbolic_util.h"

using symbolic::Expression;
using symbolic::NumericValue;

int main() {
  Expression equation = symbolic::CreateExpression("a*x + b + 0.5i");

  std::cout << "Equation: \n" << equation.to_string() << std::endl;
}

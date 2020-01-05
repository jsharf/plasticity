#include <cstdlib>

#include <iostream>
#include <set>
#include <memory>
#include <experimental/optional>

#include "symbolic/numeric_value.h"
#include "symbolic/expression.h"
#include "symbolic/symbolic_util.h"

using symbolic::Expression;
using symbolic::NumericValue;

int main() {
  Expression equation = symbolic::CreateExpression("a*x + b + 0.5i");

  std::cout << "Equation: \n" << equation.to_string() << std::endl;
}

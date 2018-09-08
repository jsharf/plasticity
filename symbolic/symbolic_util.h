#ifndef SYMBOLIC_UTIL_H
#define SYMBOLIC_UTIL_H

#include "math/geometry/dynamic_matrix.h"
#include "math/symbolic/expression.h"
#include "math/symbolic/numeric_value.h"

namespace symbolic {

Expression Sigmoid(const Expression& a);

Expression Relu(const Expression& a);

Expression Identity(const Expression& a);

// Returns log of exp with base base. Aka log(exp)/log(base).
Expression Log(NumericValue base, const Expression& exp);

// Natural logarithm.
Expression Log(const Expression& exp);

// Returns base^expression.
Expression Exp(NumericValue base, const Expression& exp);

// e^(exp).
Expression Exp(const Expression& exp);

Expression Softmax(const Matrix<Expression>& column_vector, int index);

Expression Max(const std::vector<Expression>& exprs);

Matrix<double> MapBindAndEvaluate(Matrix<symbolic::Expression> symbols,
                                  symbolic::Environment env);

symbolic::Expression Unflatten3dRow(size_t width, size_t height, size_t depth,
                                    const symbolic::Expression& i) {
  symbolic::Expression z_plane_size(symbolic::Integer(width * height));
  symbolic::Expression z_plane = i / z_plane_size;
  symbolic::Expression plane_index = i - z_plane;
  symbolic::Expression row = plane_index / width;
  return row;
}

symbolic::Expression Unflatten3dCol(size_t width, size_t height, size_t depth,
                                    const symbolic::Expression& i) {
  symbolic::Expression z_plane_size(symbolic::Integer(width * height));
  symbolic::Expression z_plane = i / z_plane_size;
  symbolic::Expression plane_index = i - z_plane;
  symbolic::Expression col = plane_index % width;
  return col;
}

symbolic::Expression Unflatten3dPlane(size_t width, size_t height, size_t depth,
                                      const symbolic::Expression& i) {
  symbolic::Expression z_plane_size(symbolic::Integer(width * height));
  symbolic::Expression z_plane = i / z_plane_size;
  return z_plane;
}

}  // namespace symbolic

#endif /* SYMBOLIC_UTIL_H */

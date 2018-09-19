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

// 3D array flattening & unflattening.

symbolic::Expression Flatten3d(size_t width, size_t height, size_t depth,
                               const symbolic::Expression& row,
                               const symbolic::Expression& col,
                               const symbolic::Expression& plane) {
  symbolic::Expression z_plane_size(symbolic::Integer(width * height));
  return z_plane_size * plane + row * width + col;
}

symbolic::Expression Unflatten3dRow(size_t width, size_t height, size_t depth,
                                    const symbolic::Expression& i) {
  symbolic::Expression z_plane_size(symbolic::Integer(width * height));
  symbolic::Expression z_plane = i / z_plane_size;
  symbolic::Expression plane_index = i - z_plane * z_plane_size;
  symbolic::Expression row = plane_index / width;
  return row;
}

symbolic::Expression Unflatten3dCol(size_t width, size_t height, size_t depth,
                                    const symbolic::Expression& i) {
  symbolic::Expression z_plane_size(symbolic::Integer(width * height));
  symbolic::Expression z_plane = i / z_plane_size;
  symbolic::Expression plane_index = i - z_plane * z_plane_size;
  symbolic::Expression col = plane_index % width;
  return col;
}

symbolic::Expression Unflatten3dPlane(size_t width, size_t height, size_t depth,
                                      const symbolic::Expression& i) {
  symbolic::Expression z_plane_size(symbolic::Integer(width * height));
  symbolic::Expression z_plane = i / z_plane_size;
  return z_plane;
}

// 2D array flattening & unflattening.

symbolic::Expression Flatten2d(size_t width, size_t height,
                               const symbolic::Expression& row,
                               const symbolic::Expression& col) {
  return symbolic::Expression(width) * row + col;
}

symbolic::Expression Unflatten2dRow(size_t width, size_t height,
                                    const symbolic::Expression& i) {
  return i / symbolic::Integer(width);
}

symbolic::Expression Unflatten2dCol(size_t width, size_t height,
                                    const symbolic::Expression& i) {
  return i % symbolic::Integer(width);
}

// LT expression
symbolic::Expression LtExpression(const symbolic::Expression& a,
                                  const symbolic::Expression& b) {
  return Expression(std::make_shared<const NotExpression>(
      symbolic::Expression(std::make_shared<GteExpression>(a, b))));
}

symbolic::Expression IfInRange(const symbolic::Expression& index,
                               const symbolic::Expression& a,
                               const symbolic::Expression& b,
                               const symbolic::Expression& then,
                               const symbolic::Expression& ifnot) {
  const symbolic::Expression gtea(
      std::make_shared<symbolic::GteExpression>(index, a));
  const symbolic::Expression ltb = LTExpression(b, index);
  const symbolic::Expression gtea_and_ltb(
      std::make_shared<symbolic::AndExpression>(gtea, ltb));
  return symbolic::Expression(
      std::make_shared<symbolic::IfExpression>(gtea_and_ltb, then, ifnot));
}

}  // namespace symbolic

#endif /* SYMBOLIC_UTIL_H */

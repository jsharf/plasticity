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

// If the input value is zero, it will have a tiny epsilon added to it to
// compute the approximated log.
Expression SafeLog(const Expression& exp);

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
                               const symbolic::Expression &row,
                               const symbolic::Expression &col,
                               const symbolic::Expression &plane);
symbolic::Expression Unflatten3dRow(size_t width, size_t height, size_t depth,
                                    const symbolic::Expression& i);
symbolic::Expression Unflatten3dCol(size_t width, size_t height, size_t depth,
                                    const symbolic::Expression& i);
symbolic::Expression Unflatten3dPlane(size_t width, size_t height, size_t depth,
                                      const symbolic::Expression& i);

// 2D array flattening & unflattening.
symbolic::Expression Flatten2d(size_t width, size_t height,
                               const symbolic::Expression& row,
                               const symbolic::Expression& col);
symbolic::Expression Unflatten2dRow(size_t width, size_t height,
                                    const symbolic::Expression& i);
symbolic::Expression Unflatten2dCol(size_t width, size_t height,
                                    const symbolic::Expression& i);

symbolic::Expression LtExpression(const symbolic::Expression& a,
                                  const symbolic::Expression& b);

symbolic::Expression IfInRange(const symbolic::Expression& index,
                               const symbolic::Expression& a,
                               const symbolic::Expression& b,
                               const symbolic::Expression& then,
                               const symbolic::Expression& ifnot);

}  // namespace symbolic

#endif /* SYMBOLIC_UTIL_H */

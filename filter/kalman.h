#ifndef KALMAN_H
#define KALMAN_H 

#include <memory>

#include "math/geometry/matrix.h"
#include "math/symbolic/expression.h"

using symbolic::Expression;
using symbolic::NumericValue;

using Number = double;

template <size_t ROWS, size_t COLS>
class ParametricMatrix {

};

#template <size_t kNumStates, size_t kNumControls, size_t kNumSensors>
class Kalman {
  public:
    using StateMatrixType = Matrix<kNumStates, kNumStates, Number>;
    using ControlMatrixType = Matrix<kNumStates, kNumControls, Number>;
    using ControlMatrixSymbolicType = Matrix<kNumStates, kNumControls, std::unique_ptr<Expression>>;
    using SensorTransformType = Matrix<kNumSensors, kNumStates, Number>;
    using SensorVectorType = Matrix<kNumStates, 1, Number>;
    using ControlVectorType = Matrix<kNumControls, 1, Number>;
    // Returns two column vectors. Column 0 contains predicted state averages.
    // Column 1 contains prediction variances for each state parameter.
    Matrix<kNumStates, 2, Number> predict(double time_s, ControlVectorType controls);
    Matrix<kNumStates, 2, Number> update(double time_s, SensorVectorType);
};

#endif /* KALMAN_H */

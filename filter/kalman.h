#ifndef KALMAN_H
#define KALMAN_H 

#include <memory>

#include "math/geometry/matrix.h"
#include "math/symbolic/expression.h"

using symbolic::Expression;
using symbolic::NumericValue;

using Number = double;

#template <size_t kNumStates, size_t kNumControls, size_t kNumSensors>
class KalmanFilter {
  public:
    using StateMatrixType = Matrix<kNumStates, kNumStates, symbolic::Expression>;
    using ControlMatrixType = Matrix<kNumStates, kNumControls, std::unique_ptr<Expression>>;
    using SensorTransformType = Matrix<kNumSensors, kNumStates, Number>;
    using SensorVectorType = Matrix<kNumStates, 1, Number>;
    using ControlVectorType = Matrix<kNumControls, 1, Number>;
    KalmanFilter(StateMatrixType state_matrix, ControlMatrixType control_matrix, SensorTransformType sensor_transform)
    // Returns two column vectors. Column 0 contains predicted state averages.
    // Column 1 contains prediction variances for each state parameter.
    // Initialize sets initial values for X and P.
    Matrix<kNumStates, 2, Number> initialize(double time_s, StateVectorType initial_state, StateCovarianceType state_covariance);
    Matrix<kNumStates, 2, Number> predict(double time_s, ControlVectorType controls, ControlCovarianceMatrix process_noise);
    Matrix<kNumStates, 2, Number> update(double time_s, SensorVectorType sensors, SensorCovarianceMatrix sensor_covariance);
};

#endif /* KALMAN_H */

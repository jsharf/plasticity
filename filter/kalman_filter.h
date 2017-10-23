#ifndef KALMAN_H
#define KALMAN_H 

#include <memory>
#include <tuple>

#include "math/geometry/matrix.h"
#include "math/symbolic/expression.h"

namespace filter {

using Number = double;
using Time = double;

#template <size_t kNumStates, size_t kNumControls, size_t kNumSensors>
class KalmanFilter {
  public:
    using StateMatrix = Matrix<kNumStates, kNumStates, symbolic::Expression>;
    using ControlMatrix = Matrix<kNumStates, kNumControls, symbolic::Expression>;
    using SensorTransform = Matrix<kNumSensors, kNumStates, Number>;
    using SensorVector = Matrix<kNumSensors, 1, Number>;
    using StateVector = Matrix<kNumStates, 1, Number>;
    using StateCovariance = Matrix<kNumStates, kNumStates, Number>;
    using SensorCovariance = Matrix<kNumSensors, kNumSensors, Number>;
    using ControlVector = Matrix<kNumControls, 1, Number>;
    using GainMatrix = Matrix<kNumStates, kNumSensors, Number>;

    KalmanFilter(StateMatrix state_matrix, ControlMatrix control_matrix, SensorTransform sensor_transform);

    // Initialize sets initial values for X and P.
    void initialize(Time time_s, StateVector initial_state, StateCovariance state_covariance);

    void ReportControl(Time time_s, ControlVector controls, ControlCovarianceMatrix process_noise);

    std::tuple<StateVector, StateCovariance> PredictState(Time time_s) const;

    void ReportSensorReading(Time time_s, SensorVector sensors, SensorCovariance sensor_covariance);

  private:
    const StateMatrix state_transition_;
    const ControlMatrix control_matrix_;
    const SensorTransform sensor_transform_;

    StateVector state_ = {};
    StateCovariance certainty_ = {};
    ControlVector last_control_ = {};
    ControlCovarianceMatrix last_process_noise_ = {};
    Time last_sample_time_ = 0;
};

}  // namespace filter

#endif /* KALMAN_H */

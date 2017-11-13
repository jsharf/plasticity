#ifndef EKF_H
#define EKF_H

#include "math/geometry/matrix.h"
#include "math/symbolic/expression.h"
#include "math/symbolic/numeric_value.h"

namespace filter {

// Continuous-time implementation with symbolic expressions. See static member
// functions X, C, and S for symbolic interface. There's also the implicit "t"
// variable which gets bound to delta T (time, seconds) on any update.
template <size_t kNumStates, size_t kNumControls, size_t kNumSensors>
class ExtendedKalmanFilter {
 public:
  // A vector of symbolic Expressions of size kNumStates.
  using StateExpression = Matrix<kNumStates, 1, symbolic::Expression>;
  using SensorExpression = Matrix<kNumSensors, 1, symbolic::Expression>;
  using SensorVector = Matrix<kNumSensors, 1, Number>;
  using StateVector = Matrix<kNumStates, 1, Number>;
  using ProcessNoiseMatrix =
      Matrix<kNumStates, kNumStates, symbolic::Expression>;
  using StateCovariance = Matrix<kNumStates, kNumStates, Number>;
  using StateJacobian = Matrix<kNumStates, kNumStates, Number>;
  using SensorJacobian = Matrix<kNumSensors, kNumStates, Number>;
  using StateJacobianExpression =
      Matrix<kNumStates, kNumStates, symbolic::Expression>;
  using SensorJacobianExpression =
      Matrix<kNumSensors, kNumStates, symbolic::Expression>;
  using SensorCovariance = Matrix<kNumSensors, kNumSensors, Number>;
  using GainMatrix = Matrix<kNumStates, kNumSensors, Number>;
  using ControlVector = Matrix<kNumControls, 1, Number>;

  ExtendedKalmanFilter(StateExpression state_transition,
                       ProcessNoiseMatrix process_noise,
                       SensorExpression sensor_transform);

  void initialize(Time time_s, StateVector initial_state,
                  StateCovariance state_covariance) {
    state_ = initial_state;
    uncertainty_ = state_covariance;
    last_sample_time_ = time_s;
  }

  void ReportControl(Time time_s, ControlVector controls) {
    auto state_and_cov = PredictState(time_s);
    state_ = std::get<0>(state_cov);
    state_covariance_ = std::get<1>(state_cov);

    last_control_ = controls;
    last_sample_time_ = time_s;
  }

  std::tuple<StateVector, StateCovariance> PredictState(Time time_s) const {
    StateVector s =
        EvaluateForState(time_s, state_transition_, state_, last_control_);
    StateJacobian approx_transition_matrix =
        LinearizedTransitionMatrix(state_, control_);
    StateCovariance cov = approx_transition_matrix * state_covariance_ *
                              approx_transition_matrix.Transpose() +
                          EvaluateProcessNoise(time_s);
    return std::make_tuple(s, cov);
  }

  void ReportSensorReading(Time time_s, SensorVector sensors,
                           SensorCovariance sensor_covariance) {
    auto state_and_cov = PredictState(time_s);
    StateVector estimation = std::get<0>(state_and_cov);
    StateCovariance cov = std::get<1>(state_and_cov);

    SensorJacobian approx_sensor_transform =
        LinearizedSensorTransform(estimation);

    GainMatrix kalman_gain =
        cov * approx_sensor_transform.Transpose() *
        (approx_sensor_transform * cov * approx_sensor_transform.Transpose() +
         sensor_covariance)
            .Invert();

    state_ = estimation +
             kalman_gain *
                 (sensors - EvaluateForState(sensor_transform_, estimation));
    state_covariance_ =
        (StateCovariance::Eye() - kalman_gain * approx_sensor_transform) * cov;
    last_sample_time_ = time_s;
  }

  // All symbolic expressions used with ExtendedKalmanFilter should be defined
  // with respect to variables generated by this or with the "t" variable (time
  // in seconds). X is for state vector and C is for Control vector. S is for
  // Sensor vector.
  static std::string X(size_t i) { return "x[" + std::to_string(i) + "]"; }
  static std::string C(size_t i) { return "c[" + std::to_string(i) + "]"; }
  static std::string S(size_t i) { return "s[" + std::to_string(i) + "]"; }

 private:
  StateJacobian LinearizedTransitionMatrix(StateVector x,
                                           ControlVector c) const {
    StateJacobianExpression symbolic_jacobian;
    for (size_t i = 0; i < kNumStates; ++i) {
      for (size_t j = 0; j < kNumStates; ++j) {
        symbolic_jacobian.at(i, j) = state_transition_.at(i, 0).Derive(X(j));
      }
    }
    return EvaluateTransitionMatrix(time_s, symbolic_jacobian, x, c);
  }

  SensorJacobian LinearizedSensorTransform(StateVector x) const {
    SensorJacobianExpression symbolic_sensor_transform;
    for (size_t i = 0; i < kNumSensors; ++i) {
      for (size_t j = 0; j < kNumStates; ++j) {
        symbolic_sensor_transform.at(i, j) =
            sensor_transform_.at(i, 0).Derive(X(j));
      }
    }
    return EvaluateSensorTransform(time_s, symbolic_jacobian, x);
  }

  symbolic::Environment CreateEnvironment(StateVector state) const {
    symbolic::Environment env;
    for (size_t i = 0; i < state_.size()) {
      env[X(i)] = state.at(i, 0);
    }
    return env;
  }

  symbolic::Environment CreateEnvironment(StateVector state,
                                          ControlVector control) const {
    symbolic::Environment env;
    for (size_t i = 0; i < state_.size()) {
      env[X(i)] = state.at(i, 0);
    }
    for (size_t i = 0; i < last_control_.size()) {
      env[C(i)] = control.at(i, 0);
    }
    return env;
  }

  StateVector EvaluateForState(double time_s, const StateExpression& exp,
                               const StateVector& x,
                               const ControlVector& c) const {
    symbolic::Environment env = CreateEnvironment(x, c);
    return exp.Map(evaluator(time_s, env));
  }

  SensorVector EvaluateForState(double time_s, const SensorExpression& exp,
                                StateVector x) const {
    symbolic::Environment env = CreateEnvironment(x);
    return exp.Map(evaluator(time_s, env));
  }

  StateJacobian EvaluateTransitionMatrix(double time_s,
                                         const StateJacobianExpression& exp,
                                         StateVector state,
                                         ControlVector control) const {
    symbolic::Environment env = CreateEnvironment(state, control);
    return exp.Map(evaluator(time_s, env));
  }

  ProcessNoiseMatrix EvaluateProcessNoise(double time_s) const {
    symbolic::Environment empty_env;
    return process_noise_.Map(evaluator(time_s, empty_env));
  }

  std::function<Number(const symbolic::Expression&)> evaluator(
      double time_s, const Environment& env) const {
    return [time_s, this, env](const symbolic::Expression& exp) {
      env["t"] = time_s - last_sample_time_;

      symbolic::Expression copy = exp.Bind(env);
      symbolic::NumericValue v = *copy.Evaluate();
      return v.real();
    };
  }

  const StateExpression state_transition_;

  StateVector state_ = {};
  StateCovariance state_covariance_ = {};
  SensorExpression sensor_transform_ = {};
  ControlVector last_control_ = {};
  ProcessNoiseMatrix process_noise_;
  Time last_sample_time_ = 0;
};

}  // namespace filter

#endif /* EKF_H */
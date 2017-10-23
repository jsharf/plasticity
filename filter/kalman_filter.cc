#include "kalman_filter.h"

namespace filter {

using symbolic::Expression;
using symbolic::NumericValue;

KalmanFilter::KalmanFilter(StateMatrix state_matrix, ControlMatrix control_matrix, SensorTransform sensor_transform) : state_transition_(state_matrix), control_matrix_(control_matrix), sensor_transform_(sensor_transform) {}

// Returns two column vectors. Column 0 contains predicted state averages.
// Column 1 contains prediction variances for each state parameter.
// Initialize sets initial values for X and P.
void KalmanFilter::initialize(Time time_s, StateVector initial_state, StateCovariance state_covariance) {
  state_ = initial_state;
  certainty_ = state_covariance;
  last_sample_time_ = time_s;
}

void ReportControl(Time time_s, ControlVector controls, ControlCovarianceMatrix process_noise) {
  expression_evaluator = [last_sample_time_](Expression exp) -> Number {
    Expression copy = exp;
    copy.bind("t", time_s - last_sample_time_);
    return *copy.Evaluate();
  }

  Matrix<kNumStates, kNumStates, Number> state_transition = state_transition_.map(expression_evaluator)

  Matrix<kNumStates, kNumControls, Number> control_matrix = control_matrix_.map(expression_evaluator);

  state_ = state_transition * state_vector + control_matrix * controls;
  certainty_ = state_transition * certainty_ * state_transition.transpose() + process_noise;

  last_control_ = controls;
  last_process_noise_ = process_noise;
  last_sample_time_ = time_s;
}

void KalmanFilter::PredictState(Time time_s) const {
  expression_evaluator = [last_sample_time_](Expression exp) -> Number {
    Expression copy = exp;
    copy.bind("t", time_s - last_sample_time_);
    return *copy.Evaluate();
  }

  Matrix<kNumStates, kNumStates, Number> state_transition = state_transition_.map(expression_evaluator)

  Matrix<kNumStates, kNumControls, Number> control_matrix = control_matrix_.map(expression_evaluator);

  StateVector estimation = state_transition * state_vector + control_matrix * last_control_;
  StateCovariance certainty = state_transition * certainty_ * state_transition.transepose() + process_noise;

  return std::make_tuple(estimation, certainty);
}

void ReportSensorReading(Time time_s, SensorVector sensors, SensorCovariance sensor_covariance) {
  auto state_and_cov = PredictState(time_s);
  StateVector estimation = state_and_cov.first;
  StateCovariance certainty = state_and_cov.second;

  GainMatrix kalman_gain = certainty * sensor_transform_.transpose() * (sensor_transform_ * certainty * sensor_transform_.transpose + sensor_covariance).Invert();

  state_ = estimation + kalman_gain * (sensors - sensor_transform_ * estimation);
  certainty_ = certainty - kalman_gain * sensor_transform_ * certainty;
  last_sample_time_ = 0;
}


}   // namespace filter

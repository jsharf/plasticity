#ifndef KALMAN_H
#define KALMAN_H

#include <memory>
#include <tuple>

#include "math/geometry/matrix.h"
#include "math/symbolic/expression.h"
#include "math/symbolic/numeric_value.h"

namespace filter {

using Number = double;
using Time = double;

template <size_t kNumStates, size_t kNumControls, size_t kNumSensors>
class KalmanFilter {
 public:
  using StateMatrix = Matrix<kNumStates, kNumStates, symbolic::Expression>;
  using ControlMatrix = Matrix<kNumStates, kNumControls, symbolic::Expression>;
  using ProcessNoiseMatrix =
      Matrix<kNumStates, kNumStates, symbolic::Expression>;
  using SensorTransform = Matrix<kNumSensors, kNumStates, Number>;
  using SensorVector = Matrix<kNumSensors, 1, Number>;
  using StateVector = Matrix<kNumStates, 1, Number>;
  using StateCovariance = Matrix<kNumStates, kNumStates, Number>;
  using SensorCovariance = Matrix<kNumSensors, kNumSensors, Number>;
  using ControlVector = Matrix<kNumControls, 1, Number>;
  using GainMatrix = Matrix<kNumStates, kNumSensors, Number>;

  KalmanFilter(StateMatrix state_matrix, ControlMatrix control_matrix,
               ProcessNoiseMatrix process_noise,
               SensorTransform sensor_transform)
      : state_transition_(state_matrix),
        control_matrix_(control_matrix),
        process_noise_(process_noise),
        sensor_transform_(sensor_transform) {}

  // Initialize sets initial values for X and P.
  void initialize(Time time_s, StateVector initial_state,
                  StateCovariance state_covariance) {
    state_ = initial_state;
    certainty_ = state_covariance;
    last_sample_time_ = time_s;
  }

  void ReportControl(Time time_s, ControlVector controls) {
    std::function<Number(const symbolic::Expression&)> expression_evaluator =
        [time_s, this](const symbolic::Expression& exp) {
          symbolic::Expression copy =
              exp.Bind("t", symbolic::NumericValue(time_s - last_sample_time_));
          symbolic::NumericValue v = *copy.Evaluate();
          return v.real();
        };

    Matrix<kNumStates, kNumStates, Number> state_transition =
        state_transition_.Map(expression_evaluator);

    Matrix<kNumStates, kNumControls, Number> control_matrix =
        control_matrix_.Map(expression_evaluator);

    Matrix<kNumStates, kNumStates, Number> process_noise =
        process_noise_.Map(expression_evaluator);

    state_ = state_transition * state_ + control_matrix * controls;
    certainty_ = state_transition * certainty_ * state_transition.Transpose() +
                 process_noise;

    last_control_ = controls;
    last_sample_time_ = time_s;
  }

  std::tuple<StateVector, StateCovariance> PredictState(Time time_s) const {
    std::function<Number(const symbolic::Expression&)> expression_evaluator =
        [time_s, this](const symbolic::Expression& exp) {
          symbolic::Expression copy =
              exp.Bind("t", symbolic::NumericValue(time_s - last_sample_time_));
          symbolic::NumericValue v = *copy.Evaluate();
          return v.real();
        };

    Matrix<kNumStates, kNumStates, Number> state_transition =
        state_transition_.Map(expression_evaluator);

    Matrix<kNumStates, kNumControls, Number> control_matrix =
        control_matrix_.Map(expression_evaluator);

    Matrix<kNumStates, kNumStates, Number> process_noise =
        process_noise_.Map(expression_evaluator);

    StateVector estimation =
        state_transition * state_ + control_matrix * last_control_;
    StateCovariance certainty =
        state_transition * certainty_ * state_transition.Transpose() +
        process_noise;

    std::function<std::string(const symbolic::Expression& exp)>
        stringifier_debug = [](const symbolic::Expression& exp) -> std::string {
      return exp.to_string();
    };

    return std::make_tuple(estimation, certainty);
  }

  void ReportSensorReading(Time time_s, SensorVector sensors,
                           SensorCovariance sensor_covariance) {
    auto state_and_cov = PredictState(time_s);
    StateVector estimation = std::get<0>(state_and_cov);
    StateCovariance certainty = std::get<1>(state_and_cov);

    GainMatrix kalman_gain =
        certainty * sensor_transform_.Transpose() *
        (sensor_transform_ * certainty * sensor_transform_.Transpose() +
         sensor_covariance)
            .Invert();

    state_ =
        estimation + kalman_gain * (sensors - sensor_transform_ * estimation);
    certainty_ = certainty - kalman_gain * sensor_transform_ * certainty;

    //std::cerr << "Sensor update: " << std::endl;
    //std::cerr << "Time gap: " << time_s - last_sample_time_ << std::endl;
    //std::cerr << "Gain: " << kalman_gain.to_string() << std::endl;
    //std::cerr << "Sensor-space prediction: "
    //          << (sensor_transform_ * estimation).to_string() << std::endl;
    //std::cerr << "Sensor transform: " << sensor_transform_.to_string();
    //std::cerr << "Sensors: " << sensors.to_string() << std::endl;
    //std::cerr << "certainty: " << certainty.to_string() << std::endl;

    last_sample_time_ = time_s;
  }

 private:
  const StateMatrix state_transition_;
  const ControlMatrix control_matrix_;
  const ProcessNoiseMatrix process_noise_;
  const SensorTransform sensor_transform_;

  StateVector state_ = {};
  StateCovariance certainty_ = {};
  ControlVector last_control_ = {};
  Time last_sample_time_ = 0;
};

}  // namespace filter

#endif /* KALMAN_H */

#include <iostream>

#include "math/filter/kalman_filter.h"
#include "math/symbolic/expression.h"

using symbolic::CreateExpression;

typedef double Number;

int main() {
  constexpr size_t kNumStates = 2;    // Position, Velocity.
  constexpr size_t kNumControls = 0;  // Throttle.
  constexpr size_t kNumSensors = 1;   // Position sensor.
  using KalmanFilter =
      filter::KalmanFilter<kNumStates, kNumControls, kNumSensors>;

  // Matrix<kNumStates, kNumStates, symbolic::Expression> state_matrix = {
  KalmanFilter::StateMatrix state_matrix = {
      {CreateExpression("1"), CreateExpression("t")},
      {CreateExpression("0"), CreateExpression("1")},
  };

  // Matrix<kNumStates, kNumControls, symbolic::Expression> control_matrix = {
  KalmanFilter::ControlMatrix control_matrix = {{}};

  KalmanFilter::ProcessNoiseMatrix process_noise = {
      {CreateExpression("0.3 * t * t * t"), CreateExpression("0.5 * t * t")},
      {CreateExpression("0.5 * t * t"), CreateExpression("t")},
  };

  // Fill noise.
  process_noise = process_noise * CreateExpression("0.00001");

  // Matrix<kNumSensors, kNumStates, Number> sensor_transform = {
  KalmanFilter::SensorTransform sensor_transform = {
      {1, 0},
  };

  KalmanFilter::SensorCovariance sensor_covariance{0.1};

  KalmanFilter simple_tank_demo(state_matrix, control_matrix, process_noise,
                                sensor_transform);

  simple_tank_demo.initialize(
      0, KalmanFilter::StateVector{{0}, {0}},
      KalmanFilter::StateCovariance{{1000, 0}, {0, 1000}});
};

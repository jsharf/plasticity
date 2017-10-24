#include <iostream>

#include "math/symbolic/expression.h"
#include "math/filter/kalman_filter.h"

using symbolic::CreateExpression;

typedef double Number;

int main() {
  constexpr size_t kNumStates = 2;  // Position, Velocity.
  constexpr size_t kNumControls = 1;  // Throttle.
  constexpr size_t kNumSensors = 1; // Position sensor.
  using KalmanFilter = filter::KalmanFilter<kNumStates, kNumControls, kNumSensors>;
  
  // Matrix<kNumStates, kNumStates, symbolic::Expression> state_matrix = {
  KalmanFilter::StateMatrix state_matrix = {
    {CreateExpression("1"), CreateExpression("t")},
    {CreateExpression("0"), CreateExpression("1")},
  };

  // Matrix<kNumStates, kNumControls, symbolic::Expression> control_matrix = {
  KalmanFilter::ControlMatrix control_matrix = {
    {CreateExpression("t * t * 0.5")},
    {CreateExpression("t")},
  };

  KalmanFilter::ProcessNoiseMatrix process_noise = {
    {CreateExpression("0.05"), CreateExpression("0")},
    {CreateExpression("0"), CreateExpression("0.05")},
  };

  // Matrix<kNumSensors, kNumStates, Number> sensor_transform = {
  KalmanFilter::SensorTransform sensor_transform = {{
      3.3 * 0.000000001,  // 1/c.
      0
    },
  };
   
  KalmanFilter simple_cart_demo(state_matrix, control_matrix, process_noise, sensor_transform); 
};

#include <array>
#include <chrono>
#include <iostream>
#include <sstream>
#include <string>

#include "math/filter/kalman_filter.h"
#include "math/symbolic/expression.h"

#include "graphics/camera_3d.h"
#include "graphics/perspective_camera.h"
#include "graphics/scene.h"
#include "graphics/sdl_canvas.h"
#include "graphics/types.h"
#include "math/geometry/matrix4.h"
#include "math/geometry/vector.h"

#include <SDL.h>

using symbolic::CreateExpression;

typedef double Number;

using Quad = std::array<Vector3, 4>;

class ArrowScene : public Scene {
 public:
  std::vector<Triangle3> GetPrimitives() override {
    std::vector<Triangle3> triangles;
    for (size_t i = 0; i < arrow_.size(); ++i) {
      triangles.push_back(
          OffsetTriangle(offset_, RotateTriangle(orientation_, arrow_[i])));
    }
    return triangles;
  }

  void set_orientation(Quaternion orientation) {
    orientation_ = Matrix4::Rot(orientation);
  }

  void set_orientation(double pitch, double roll) {
    orientation_ = Matrix4::RotI(pitch) * Matrix4::RotJ(roll);
  }

  void set_offset(double x, double y, double z) { offset_ = Vector3(x, y, z); }

 private:
  Triangle3 RotateTriangle(Matrix4 rot, Triangle3 tri) {
    Triangle3 result;
    for (size_t i = 0; i < tri.size(); ++i) {
      result[i] = rot * tri[i];
    }
    return result;
  }

  Triangle3 OffsetTriangle(Vector3 offset, Triangle3 tri) {
    Triangle3 result;
    for (size_t i = 0; i < tri.size(); ++i) {
      result[i] = offset + tri[i];
    }
    return result;
  }

  std::array<Triangle3, 6> arrow_ = {{
      {{Vector3(0, -0.5, -0.3), Vector3(0, -0.5, 0.3), Vector3(0, 0.5, 0)}},
      {{Vector3(0, -0.5, 0.3), Vector3(2, 0, 0), Vector3(0, 0.5, 0)}},
      {{Vector3(0, -0.5, -0.3), Vector3(2, 0, 0), Vector3(0, -0.5, 0.3)}},
      {{Vector3(0, 0.5, 0), Vector3(2, 0, 0), Vector3(0, -0.5, -0.3)}},
  }};

  Matrix4 orientation_ = Matrix4::eye();
  Vector3 offset_ = {};
};

class Timer {
 public:
  Timer() {}

  void Start() { StartAt(0); }

  bool started() const { return started_; }

  void StartAt(double time_s) {
    start_ = std::chrono::system_clock::now();
    offset_ = std::chrono::duration<double>(time_s);
    started_ = true;
  }

  double seconds() {
    return (std::chrono::duration_cast<
                std::chrono::duration<double, std::ratio<1>>>(
                std::chrono::system_clock::now() - start_) +
            offset_)
        .count();
  }

 private:
  std::chrono::system_clock::time_point start_ = {};
  std::chrono::duration<double> offset_ = {};
  bool started_ = false;
};

int main() {
  constexpr size_t kNumStates = 6;    // Position, Velocity.
  constexpr size_t kNumControls = 0;  // Throttle.
  constexpr size_t kNumSensors = 6;   // Position sensor.
  using KalmanFilter =
      filter::KalmanFilter<kNumStates, kNumControls, kNumSensors>;

  // Matrix<kNumStates, kNumStates, symbolic::Expression> state_matrix = {
  symbolic::Expression zero = CreateExpression("0");
  symbolic::Expression one = CreateExpression("1");
  symbolic::Expression t = CreateExpression("t");

  // State vector is Transpose([degx, degy, degz, degx' degy' degz'])
  KalmanFilter::StateMatrix state_matrix = {
      {one, zero, zero, t, zero, zero},    {zero, one, zero, zero, t, zero},
      {zero, zero, one, zero, zero, t},    {one, zero, zero, zero, zero, zero},
      {zero, one, zero, zero, zero, zero}, {zero, zero, one, zero, zero, zero},
  };

  // Matrix<kNumStates, kNumControls, symbolic::Expression> control_matrix = {
  KalmanFilter::ControlMatrix control_matrix = {};

  // Noise is small but grows exponentially with time.
  auto exponential_noise = CreateExpression("0.05 * t * t * t");
  KalmanFilter::ProcessNoiseMatrix process_noise;
  for (size_t i = 0; i < kNumStates; ++i) {
    process_noise.at(i, i) = exponential_noise;
  }

  // Matrix<kNumSensors, kNumStates, Number> sensor_transform = {
  // Sensor vector is Transpose([Ax, Ay, Az, Gx, Gy, Gz])
  //
  // This uses the small angle theorem to approximate sin(theta) for converting
  // orientation to acceleration readings. Not very accurate.
  KalmanFilter::SensorTransform sensor_transform = {
      {1, 0, 0, 0},
      {0, 1, 0, 0, 0},
      {0, 0, 1, 0, 0, 0},
      {0, 0, 0, (180 / (3.1415 * 0.00875)), 0, 0},
      {0, 0, 0, 0, (180 / (3.1415 * 0.00875)), 0},
      {0, 0, 0, 0, 0, 180 / (3.1415 * 0.00875)},
  };
  std::cout << "hi" << std::endl;

  KalmanFilter simple_imu_demo(state_matrix, control_matrix, process_noise,
                               sensor_transform);
  std::cout << "hi" << std::endl;

  // Initialize at state = 0. We don't know actual initial condition, so make
  // the covariance super large and the sensor will pick up the slack in future
  // updates.
  simple_imu_demo.initialize(0, KalmanFilter::StateVector{},
                             KalmanFilter::StateCovariance::Eye() * 100);

  std::cout << "Done init filter." << std::endl;

  const int width = 320;
  const int height = 240;
  SdlCanvas raw_canvas(width, height);
  SdlCanvas canvas(width, height);
  ArrowScene raw_visualize, visualize;
  PerspectiveCamera camera;
  PerspectiveCamera raw_camera;
  camera.SetPosition(0, -2, 0);
  camera.SetOrientation(Matrix4::RotI(M_PI / 2.0));
  raw_camera.SetPosition(0, -2, 0);
  raw_camera.SetOrientation(Matrix4::RotI(M_PI / 2.0));
  SDL_Event e;

  Timer timer;

  double last_frame = 0;
  double frame_rate = 30;

  for (std::string line; std::getline(std::cin, line);) {
    SDL_PollEvent(&e);
    if (e.type == SDL_KEYDOWN && e.key.keysym.sym == SDLK_ESCAPE) break;
    if ((timer.started()) &&
        (timer.seconds() - last_frame) > (1 / frame_rate)) {
      canvas.Clear();
      raw_canvas.Clear();
      raw_camera.CaptureWorld(&raw_visualize, &raw_canvas);
      camera.CaptureWorld(&visualize, &canvas);
      canvas.Render();
      raw_canvas.Render();
      last_frame = timer.seconds();
    }

    std::stringstream sample(line);
    double dthetax = 0, dthetay = 0, dthetaz = 0;
    double ax = 0, ay = 0, az = 0;
    double time = 0;

    scanf("%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t", &dthetax, &dthetay, &dthetaz,
          &ax, &ay, &az, &time);

    std::cout << "ax: " << ax << ", ay: " << ay << ", az: " << az << std::endl;

    if (!timer.started()) {
      timer.StartAt(time);
    }

    const Vector3 k = Vector3(0, 0, 1);
    const Vector3 acceleration = Vector3(ax, ay, az).Normalize();

    Quaternion orientation(k.Cross(acceleration), k.AngleTo(acceleration));

    raw_visualize.set_orientation(orientation);

    double pitch = atan2(acceleration.j,
                         sqrt(pow(acceleration.i, 2) + pow(acceleration.k, 2)));
    double roll = atan2(-acceleration.i, acceleration.k);

    simple_imu_demo.ReportSensorReading(
        time,
        KalmanFilter::SensorVector{
            {roll}, {pitch}, {0}, {dthetax}, {dthetay}, {dthetaz}},
        {
            {0.1, 0, 0, 0, 0, 0},
            {0, 0.1, 0, 0, 0, 0},
            {0, 0, 1000, 0, 0, 0},
            {0, 0, 0, 0.05, 0, 0},
            {0, 0, 0, 0, 0.05, 0},
            {0, 0, 0, 0, 0, 0.05},
        });

    auto result = simple_imu_demo.PredictState(time);
    KalmanFilter::StateVector prediction = std::get<0>(result);
    std::cout << prediction.to_string() << std::endl;
    visualize.set_orientation(prediction.at(0, 0), prediction.at(1, 0));
  }
  return 0;
};

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

class BoxScene : public Scene {
 public:
  std::vector<Triangle3> GetPrimitives() override {
    std::vector<Triangle3> triangles;
    for (size_t i = 0; i < box_.size(); ++i) {
      auto quadtris = QuadToTriangles(box_[i]);
      for (const Triangle3& triangle : quadtris) {
        triangles.push_back(OffsetTriangle(offset_, RotateTriangle(orientation_, triangle)));
      }
    }
    return triangles;
  }

  void set_orientation(double x, double y, double z) {
    orientation_ = Matrix4::RotI(x) * Matrix4::RotJ(y) * Matrix4::RotK(z);
  }

  void set_offset(double x, double y, double z) {
    offset_ = Vector3(x, y, z);
  }

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

  static std::vector<Triangle3> QuadToTriangles(Quad quad) {
    Triangle3 a;
    Triangle3 b;
    std::vector<Triangle3> result;
    std::get<0>(a) = std::get<0>(quad);
    std::get<1>(a) = std::get<1>(quad);
    std::get<2>(a) = std::get<3>(quad);
    std::get<0>(b) = std::get<1>(quad);
    std::get<1>(b) = std::get<2>(quad);
    std::get<2>(b) = std::get<3>(quad);
    result.push_back(a);
    result.push_back(b);
    return result;
  }

  std::array<Quad, 6> box_ = {{
      {{Vector3(0, 0, 0), Vector3(1, 0, 0), Vector3(1, 1, 0),
        Vector3(0, 1, 0)}},
      {{Vector3(0, 0, 0), Vector3(1, 0, 0), Vector3(1, 0, 1),
        Vector3(0, 0, 1)}},
      {{Vector3(0, 0, 0), Vector3(0, 1, 0), Vector3(0, 1, 1),
        Vector3(0, 0, 1)}},
      {{Vector3(0, 0, 1), Vector3(1, 0, 1), Vector3(1, 1, 1),
        Vector3(0, 1, 1)}},
      {{Vector3(0, 1, 0), Vector3(1, 1, 0), Vector3(1, 1, 1),
        Vector3(0, 1, 1)}},
      {{Vector3(1, 0, 0), Vector3(1, 1, 0), Vector3(1, 1, 1),
        Vector3(1, 0, 1)}},
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
    return (std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1>>>(std::chrono::system_clock::now() - start_) + offset_).count();
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

  auto expressifier = std::function<symbolic::Expression(const double&)>(
      [](const double& value) -> symbolic::Expression {
        return CreateExpression(std::to_string(value));
      });

  Matrix<kNumStates, kNumStates, double> const_process_noise = {
    {0.05, 0, 0, 0, 0, 0},
    {0, 0.05, 0, 0, 0, 0},
    {0, 0, 0.05, 0, 0, 0},
    {0, 0, 0, 1, 0, 0},
    {0, 0, 0, 0, 1, 0},
    {0, 0, 0, 0, 0, 1},
  };

  KalmanFilter::ProcessNoiseMatrix process_noise =
      const_process_noise.Map(expressifier);

  // Matrix<kNumSensors, kNumStates, Number> sensor_transform = {
  // Sensor vector is Transpose([Ax, Ay, Az, Gx, Gy, Gz])
  //
  // This uses the small angle theorem to approximate sin(theta) for converting
  // orientation to acceleration readings. Not very accurate.
  KalmanFilter::SensorTransform sensor_transform = {
      {227.0 / 3.1415, 0, 0, 0},
      {0, 277.0 / 3.1415, 0, 0, 0},
      {0, 0, 277.0 / 3.1415, 0, 0, 0},
      {0, 0, 0, (180 / (3.1415 * 0.00875)), 0, 0},
      {0, 0, 0, 0, (180 / (3.1415 * 0.00875)), 0},
      {0, 0, 0, 0, 0, 180 / (3.1415 * 0.00875)},
  };

  KalmanFilter simple_imu_demo(state_matrix, control_matrix, process_noise,
                               sensor_transform);

  // Initialize at state = 0, cov = Identity matrix.
  simple_imu_demo.initialize(0, KalmanFilter::StateVector{},
                             KalmanFilter::StateCovariance::Eye() * 0.5);

  //  std::cout << line << std::endl;
  //}
  const int width = 640;
  const int height = 480;
  SdlCanvas canvas(width, height);
  BoxScene box;
  PerspectiveCamera camera;
  camera.SetPosition(0, -2, 1);
  camera.SetOrientation(Matrix4::RotI(M_PI / 2.0));
  SDL_Event e;

  Timer timer;
  for (std::string line; std::getline(std::cin, line);) {
    SDL_PollEvent(&e);
    if (e.type == SDL_KEYDOWN && e.key.keysym.sym == SDLK_ESCAPE) break;
    // Transformation matrix: each point rotates around origin 2PI/second.
    canvas.Clear();
    camera.CaptureWorld(&box, &canvas);
    canvas.Render();
    SDL_Delay(1);

    std::stringstream sample(line);
    double dthetax = 0, dthetay = 0, dthetaz = 0;
    double ax = 0, ay = 0, az = 0;
    double time = 0;

    scanf("%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t", &dthetax, &dthetay, &dthetaz, &ax,
          &ay, &az, &time);

    if (!timer.started()) {
      timer.StartAt((double)time);
    }

    simple_imu_demo.ReportSensorReading(
        time,
        KalmanFilter::SensorVector{
            {ax}, {ay}, {az}, {dthetax}, {dthetay}, {dthetaz}},
        KalmanFilter::SensorCovariance::Eye() * 0.85);

    if (timer.started()) {
      auto result =
          simple_imu_demo.PredictState(timer.seconds());
      KalmanFilter::StateVector prediction = std::get<0>(result);
      std::cout << prediction.to_string() << std::endl;
      box.set_offset(prediction.at(0, 0), prediction.at(1, 0), prediction.at(2, 0));
    }
  }
};

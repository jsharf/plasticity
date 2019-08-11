#ifndef BUFFER_H
#define BUFFER_H
namespace compute {

class Buffer {
 public:
  enum Location {
    CPU = 0,
    GPU,
  };

  virtual void MoveToCpu() = 0;
  virtual void MoveToGpu() = 0;
  virtual Location GetBufferLocation() = 0;
  virtual size_t size() const = 0;
  virtual void resize(
      size_t new_size,
      double default_value = std::numeric_limits<double>::quiet_NaN()) = 0;

  virtual double &operator[](size_t index) = 0;
  virtual const double &operator[](size_t index) const = 0;
};

}  // namespace compute

#endif  // BUFFER_H

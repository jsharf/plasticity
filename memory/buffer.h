

namespace memory {

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

  virtual double& operator[](size_t index) = 0;
  virtual const double& operator[](size_t index) const = 0;
};

}  // namespace memory

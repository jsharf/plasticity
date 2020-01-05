#include <memory>
#include <string>
#include <vector>

namespace compute {

class CommandQueue {
  public:
    virtual void enqueueKernel(const std::string& kernel_name, int global_size, int workgroup_size, std::unique_ptr<std::vector<compute::ClBuffer>> arguments) = 0;
    virtual void finish() = 0;
  private:
};

}  // namespace compute

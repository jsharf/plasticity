#ifndef NORMAL_H
#define NORMAL_H
#include <random>

namespace stats {

using Number = double;

class Normal {
 public:
  Normal(Number mean, Number variance)
      : gen_(rd_()), dist_(mean, sqrt(variance)) {}
  Number mean() const { return dist_.mean(); }
  Number variance() const { return dist_.stddev() * dist_.stddev(); }
  Number stddev() const { return dist_.stddev(); }
  // The product of two normally distributed random variables is not normal.
  // This is not multiplying two random variables. However the product of two
  // Normal PDFs is also a normal PDF.
  Normal operator*(const Normal& rhs) const {
    double combined_mean = (mean() * rhs.variance() + rhs.mean() * variance()) /
                           (variance() + rhs.variance());
    double combined_variance =
        (variance() * rhs.variance()) / (variance() + rhs.variance());
    return Normal(combined_mean, combined_variance);
  }
  Normal operator+(const Normal& rhs) const {
    return Normal(mean() + rhs.mean(), variance() + rhs.variance());
  }

  Number sample() { return dist_(gen_); }

 private:
  std::random_device rd_;
  std::mt19937 gen_;
  std::normal_distribution<> dist_;
};

}  // namespace stats

#endif /* NORMAL_H */

#ifndef SYMBOL_GENERATOR_H
#define SYMBOL_GENERATOR_H

#include "math/symbolic/expression.h"

#include <map>
#include <utility>

namespace nnet {

class SymbolGenerator {
 public:
  // Feed-forward layer weights.
  virtual std::string W(size_t layer, size_t node, size_t edge) = 0;
  // Convolution layer weights.
  virtual std::string W(size_t layer, size_t filter, size_t x, size_t y,
                        size_t z) = 0;
  // Convolution layer bias weights.
  virtual std::string W(size_t layer, size_t filter) = 0;

  virtual std::string I(size_t i) const = 0;
  virtual std::string O(size_t i) const = 0;

  // Residual gradients for back propagation.
  virtual std::string GRADIENT(size_t i) const {
    return "GRADIENT[" + std::to_string(i) + "]";
  }

  virtual ~SymbolGenerator() {}
};

// This class generates symbol names for neural network values. Since these
// will be used for codegen for opencl, the symbols are all one-dimensional
// indices into arrays.
class FlatWeightSymbolGenerator : public SymbolGenerator {
 public:
  // Feed-forward layer weights.
  virtual std::string W(size_t layer, size_t node, size_t edge) {
    auto tuple = std::make_tuple(layer, node, edge);
    if (ff_weight_index_.count(tuple) == 0) {
      ff_weight_index_[tuple] = weight_count_;
      ff_rev_weight_index_[weight_count_] = tuple;
      weight_count_++;
    }
    return "W[" + std::to_string(ff_weight_index_[tuple]) + "]";
  }

  // Convolution layer weights.
  virtual std::string W(size_t layer, size_t filter, size_t x, size_t y,
                        size_t z) {
    auto tuple = std::make_tuple(layer, filter, x, y, z);
    if (conv_weight_index_.count(tuple) == 0) {
      conv_weight_index_[tuple] = weight_count_;
      conv_rev_weight_index_[weight_count_] = tuple;
      weight_count_++;
    }
    return "W[" + std::to_string(conv_weight_index_[tuple]) + "]";
  }

  // Convolution layer bias weights.
  virtual std::string W(size_t layer, size_t filter) {
    auto tuple = std::make_tuple(layer, filter);
    if (conv_bias_weight_index_.count(tuple) == 0) {
      conv_bias_weight_index_[tuple] = weight_count_;
      conv_bias_rev_weight_index_[weight_count_] = tuple;
      weight_count_++;
    }
    return "W[" + std::to_string(conv_bias_weight_index_[tuple]) + "]";
  }

  virtual std::string W(size_t i) const {
    return "W[" + std::to_string(i) + "]";
  }
  virtual std::string I(size_t i) const {
    return "I[" + std::to_string(i) + "]";
  }
  virtual std::string O(size_t i) const {
    return "O[" + std::to_string(i) + "]";
  }

  size_t NumberWeights() const { return weight_count_; }

 private:
  // Mapping from <layer, node, edge> -> int. This lets each weight have a
  // single unique index.
  std::map<std::tuple<int, int, int>, int> ff_weight_index_;
  // Reverse mapping.
  std::map<int, std::tuple<int, int, int>> ff_rev_weight_index_;

  // Mapping from <layer, filter, x, y, z> -> int. This lets each weight have
  // a single unique index.
  std::map<std::tuple<int, int, int, int, int>, int> conv_weight_index_;
  // Reverse mapping.
  std::map<int, std::tuple<int, int, int, int, int>> conv_rev_weight_index_;

  // Mapping from <layer, filter> -> int. This lets each weight have a
  // single unique index.
  std::map<std::tuple<int, int>, int> conv_bias_weight_index_;
  // Reverse mapping.
  std::map<int, std::tuple<int, int>> conv_bias_rev_weight_index_;

  size_t weight_count_ = 0;
};

}  // namespace nnet

#endif /* SYMBOL_GENERATOR_H */

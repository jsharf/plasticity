#ifndef SYMBOL_GENERATOR_H
#define SYMBOL_GENERATOR_H

#include "math/nnet/layer_dimensions.h"
#include "math/nnet/layer_impl.h"
#include "math/symbolic/expression.h"

#include <map>
#include <unordered_map>
#include <utility>

namespace nnet {

class SymbolGenerator {
 public:
  std::string I(size_t i) const {
    return "I[" + std::to_string(i) + "]";
  }
  std::string O(size_t i) const {
    return "O[" + std::to_string(i) + "]";
  }

  // Residual gradients for back propagation.
  std::string GRADIENT(size_t i) const {
    return "GRADIENT[" + std::to_string(i) + "]";
  }
};

struct DenseWeightAddress {
  // If is_bias, edge value is invalid.
  bool is_bias = false;

  size_t node = 0;
  size_t edge = 0;

  bool operator==(const DenseWeightAddress& rhs) const {
    if (is_bias) {
      return node == rhs.node;
    } else {
      return (node == rhs.node) && (edge == rhs.edge);
    }
  }

  // This is used to make hashing DenseWeightAddress simpler.
  std::string to_string() const {
    if (is_bias) {
      return "{\n\tnode: " + std::to_string(node) + "\n}";
    } else {
      return "{\n\tnode: " + std::to_string(node) + "\n\tedge: " +
             std::to_string(node) + "\n}";
    }
  }

  DenseWeightAddress(size_t p_node) : is_bias(true), node(p_node) {}
  DenseWeightAddress(size_t p_node, size_t p_edge)
      : is_bias(false), node(p_node), edge(p_edge) {}
};

struct DenseWeightAddressHash {
  std::hash<std::string> hasher;
  size_t operator()(const DenseWeightAddress& obj) const {
    return hasher(obj.to_string());
  }
};

class DenseSymbolGenerator {
  public:
    explicit DenseSymbolGenerator(Dimensions dimensions) {
      size_t index = 0;
      for (size_t node = 0; node < dimensions.num_outputs; ++node) {
        for (size_t edge = 0; edge < dimensions.num_inputs; ++edge) {
          DenseWeightAddress addr(node, edge);
          weight_addr_to_index_[addr] = index;
          index++;
        }
        // Bias weight.
        DenseWeightAddress addr(node);
        weight_addr_to_index_[addr] = index;
        index++;
      }

      weights_.resize(weight_addr_to_index_.size());
      for (const auto& pair : weight_addr_to_index_) {
        int index = pair.second;
        DenseWeightAddress addr = pair.first;
        if (addr.is_bias) {
          weights_[index] = W(addr.node);
        } else {
          weights_[index] = W(addr.node, addr.edge);
        }
      }
    }
    std::string W(size_t node, size_t edge) const {
      return "W[" + std::to_string(weight_addr_to_index_.at(DenseWeightAddress(node, edge))) + "]";
    }
    // Used for bias weight for a given output node.
    std::string W(size_t node) const {
      return "W[" + std::to_string(weight_addr_to_index_.at(DenseWeightAddress(node))) + "]";
    }
    const std::vector<std::string>& weights() const {
      return weights_;
    }

   private:
    std::unordered_map<DenseWeightAddress, int, DenseWeightAddressHash>
        weight_addr_to_index_;
    std::vector<string> weights_;
};

struct ConvWeightAddress {
  // If is_bias, x, y, and z are invalid.
  bool is_bias = false;

  // Which filter this is a weight in.
  size_t filter = 0;

  // Weight's location within the filter.
  size_t x = 0;
  size_t y = 0;
  size_t z = 0;

  bool operator==(const ConvWeightAddress& rhs) const {
    if (is_bias) {
      return filter == rhs.filter;
    } else {
      return (filter == rhs.filter) && (x == rhs.x) && (y == rhs.y) && (z == rhs.z);
    }
  }

  std::string to_string() const {
    if (is_bias) {
      return "{\n\tfilter: " + std::to_string(filter) + "\n}";
    } else {
      return "{\n\tfilter: " + std::to_string(filter) + "\n\tx: " +
             std::to_string(x) + "\n\ty: " + std::to_string(y) + "\n\tz: " +
             std::to_string(z) + "\n}";
    }
  }

  ConvWeightAddress(size_t p_filter)
      : is_bias(true), filter(p_filter) {}
  ConvWeightAddress(size_t p_filter, size_t p_x, size_t p_y, size_t p_z) : is_bias(false), filter(p_filter), x(p_x), y(p_y), z(p_z) {}
};

struct ConvWeightAddressHash {
  std::hash<std::string> hasher;
  size_t operator()(const ConvWeightAddress& obj) const {
    return hasher(obj.to_string());
  }
};

class ConvSymbolGenerator {
  public:
   explicit ConvSymbolGenerator(FilterParams filters) {
     size_t index = 0;
     for (size_t filter_no = 0; filter_no < filters.num_filters; ++filter_no) {
       for (size_t x = 0; x < filters.width; ++x) {
         for (size_t y = 0; y < filters.height; ++y) {
           for (size_t z = 0; z < filters.depth; ++z) {
             ConvWeightAddress addr(filter_no, x, y, z);
             weight_addr_to_index_[addr] = index;
             index++;
           }
         }
       }
       // Bias.
       ConvWeightAddress addr(filter_no);
       weight_addr_to_index_[addr] = index;
       index++;
     }

     weights_.resize(weight_addr_to_index_.size());
     for (const auto& pair : weight_addr_to_index_) {
       int index = pair.second;
       ConvWeightAddress addr = pair.first;
       if (addr.is_bias) {
         weights_[index] = W(addr.filter);
       } else {
         weights_[index] = W(addr.filter, addr.x, addr.y, addr.z);
       }
     }
  }

  // Convolution layer weights.
  std::string W(size_t filter, size_t x, size_t y, size_t z) const {
    ConvWeightAddress addr(filter, x, y, z);
    return "W["+std::to_string(weight_addr_to_index_.at(addr))+"]";
  }

  // Convolution layer bias weights.
  std::string W(size_t filter) const {
    ConvWeightAddress addr(filter);
    return "W["+std::to_string(weight_addr_to_index_.at(addr))+"]";
  }

  const std::vector<std::string>& weights() const {
    return weights_;
  }
  private:
    std::unordered_map<ConvWeightAddress, int, ConvWeightAddressHash> weight_addr_to_index_;
    std::vector<std::string> weights_;
};

// This class generates symbol names for neural network values. Since these
// will be used for codegen for opencl, the symbols are all one-dimensional
// indices into arrays.
class FlatWeightSymbolGenerator {
 public:
  // Fully connected layer weights.
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

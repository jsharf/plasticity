#ifndef DYNAMIC_MATRIX_H
#define DYNAMIC_MATRIX_H

#include <functional>
#include <iostream>
#include <string>
#include <sstream>
#include <tuple>
#include <vector>

using std::string;

template <typename T>
class Matrix {
 public:
  Matrix(std::initializer_list<std::initializer_list<T>> values) {
    size_t i = 0;
    size_t j = 0;

    resize(values.size(), values.begin()->size());

    for (auto list : values) {
      j = 0;
      for (auto value : list) {
        at(i, j) = value;
        j++;
      }
      i++;
    }
  }

  Matrix(size_t rows, size_t cols) {
    resize(rows, cols);
  }

  Matrix(const Matrix<T>& other) : data_(other.data_) {}

  explicit Matrix(size_t rows, size_t cols, T value) {
    resize(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
      for (size_t j = 0; j < cols; ++j) {
        at(i, j) = value;
      }
    }
  }

  std::tuple<size_t, size_t> size() const {
    return std::make_tuple(data_.size(), data_[0].size());
  }

  void resize(size_t rows, size_t cols) {
    data_.resize(rows);
    for (size_t i = 0; i < rows; ++i) {
      data_[i].resize(cols);
    }
  }

  static constexpr Matrix<T> Eye(size_t rows, size_t cols) {
    Matrix<T> result(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
      result.at(i, i) = 1;
    }
    return result;
  }

  constexpr Matrix<T> operator*(T n) const {
    auto dim = size();
    size_t rows = std::get<0>(dim);
    size_t cols = std::get<1>(dim);
    Matrix<T> res;
    for (size_t i = 0; i < rows; ++i) {
      for (size_t j = 0; j < cols; ++j) {
        res.at(i, j) = at(i, j) * n;
      }
    }
    return res;
  }

  constexpr Matrix<T> operator+(Matrix<T> rhs) const {
    if (size() != rhs.size()) {
      auto rhsdim = rhs.size();
      size_t rhsrows = std::get<0>(rhsdim);
      size_t rhscols = std::get<1>(rhsdim);
      std::cerr << "Error, adding matrices of different dimensions:"
                << "(" << rhsrows << ", " << rhscols << ")"
                << std::endl;
      std::exit(1);
    }
    auto dim = size();
    size_t rows = std::get<0>(dim);
    size_t cols = std::get<1>(dim);
    Matrix<T> res;
    for (size_t i = 0; i < rows; ++i) {
      for (size_t j = 0; j < cols; ++j) {
        res.at(i, j) = at(i, j) + rhs.at(i, j);
      }
    }
    return res;
  }

  constexpr Matrix<T> operator-(Matrix<T> rhs) const {
    Matrix<T> res;
    auto dim = size();
    size_t rows = std::get<0>(dim);
    size_t cols = std::get<1>(dim);
    for (size_t i = 0; i < rows; ++i) {
      for (size_t j = 0; j < cols; ++j) {
        res.at(i, j) = at(i, j) - rhs.at(i, j);
      }
    }
    return res;
  }

  constexpr const T& at(size_t i, size_t j) const { return data_.at(i).at(j); }
  constexpr T& at(size_t i, size_t j) { return data_.at(i).at(j); }

  // Return value is a std::pair of Matrices <lower, upper>.
  constexpr std::pair<Matrix<T>, Matrix<T>> LUDecomp() {
    auto dim = size();
    size_t rows = std::get<0>(dim);
    size_t cols = std::get<1>(dim);
    if (rows != cols) {
      std::cerr << "Warning: LUDecomp requested for non-square matrix."
                << std::endl;
    }
    Matrix<T> lower(rows, cols), upper(rows, cols);

    for (size_t i = 0; i < rows; ++i) {
      upper.at(i, i) = 1;
    }

    for (size_t j = 0; j < rows; ++j) {
      for (size_t i = j; i < rows; ++i) {
        T sum = 0;
        for (size_t k = 0; k < j; ++k) {
          sum = sum + lower.at(i, k) * upper.at(k, j);
        }
        lower.at(i, j) = at(i, j) - sum;
      }

      for (size_t i = j; i < rows; ++i) {
        T sum = 0;
        for (size_t k = 0; k < j; ++k) {
          sum = sum + lower.at(j, k) * upper.at(k, i);
        }
        if (lower.at(j, j) == 0) {
          std::cerr << "det(lower) close to 0!\n Can't divide by 0...\n"
                    << std::endl;
        }
        upper.at(j, i) = (at(j, i) - sum) / lower.at(j, j);
      }
    }

    return std::make_pair(lower, upper);
  }

  constexpr Matrix<T> LUSolve(Matrix<T> b) {
    auto dim = size();
    size_t rows = std::get<0>(dim);

    auto bdim = b.size();
    if ((std::get<0>(bdim) != rows) || (std::get<1>(bdim) != 1)) {
      std::cerr
          << "Warning, Matrix b passed to LUSolve is of incorrect dimension: "
          << "(" << std::get<0>(bdim) << ", " << std::get<1>(bdim) << ")"
          << std::endl;
      std::exit(1);
    }

    Matrix<T> d(rows, 1);
    Matrix<T> x(rows, 1);
    auto matpair = LUDecomp();
    auto& lower = matpair.first;
    auto& upper = matpair.second;
    d.at(0, 0) = b.at(0, 0) / lower.at(0, 0);
    for (size_t i = 1; i < rows; ++i) {
      T sum = 0;
      for (size_t j = 0; j < i; ++j) {
        sum += lower.at(i, j) * d.at(j, 0);
      }
      d.at(i, 0) = (b.at(i, 0) - sum) / lower.at(i, i);
    }

    x.at(rows - 1, 0) = d.at(rows - 1, 0);
    for (int i = rows - 2; i >= 0; --i) {
      T sum = 0;
      for (size_t j = i + 1; j < rows; ++j) {
        sum += upper.at(i, j) * x.at(j, 0);
      }
      x.at(i, 0) = d.at(i, 0) - sum;
    }

    return x;
  }

  constexpr Matrix<T> Invert() {
    auto dim = size();
    size_t rows = std::get<0>(dim);
    size_t cols = std::get<1>(dim);

    std::vector<Matrix<T>> columns(cols, Matrix<T>(rows, 1));
    for (size_t i = 0; i < cols; ++i) {
      columns[i].at(i, 0) = 1;
      columns[i] = LUSolve(columns[i]);
    }

    Matrix<T> result(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
      for (size_t j = 0; j < cols; ++j) {
        result.at(i, j) = columns[j].at(i, 0);
      }
    }
    return result;
  }

  // Matrix multiplication is only possible if # COLS of LHS = # ROWS of RHS
  Matrix<T> operator*(Matrix<T> rhs) const {
    auto dim = size();
    size_t rows = std::get<0>(dim);
    size_t cols = std::get<1>(dim);
    size_t rhsrows = std::get<1>(rhs.size());
    size_t rhscols = std::get<1>(rhs.size());

    if (rhsrows != rows) {
      std::cerr << "Matrix passed to operator * has incorrect dimension, "
                   "cannot multiply: "
                << "(" << rhsrows << ", " << rhscols << ")"
                << std::endl;
      std::exit(1);
    }

    Matrix<T> result(rows, rhscols);
    for (size_t i = 0; i < rows; ++i) {
      for (size_t j = 0; j < rhscols; ++j) {
        T kSum = 0;
        for (size_t k = 0; k < cols; ++k) {
          kSum = kSum + at(i, k) * rhs.at(k, j);
        }
        result.at(i, j) = kSum;
      }
    }
    return result;
  }

  Matrix<T> Transpose() const {
    auto dim = size();
    size_t rows = std::get<0>(dim);
    size_t cols = std::get<1>(dim);
    Matrix<T> result(cols, rows);
    for (size_t i = 0; i < rows; ++i) {
      for (size_t j = 0; j < cols; ++j) {
        result.at(j, i) = at(i, j);
      }
    }
    return result;
  }

  template <typename ReturnType>
  Matrix<ReturnType> Map(
      const std::function<ReturnType(const T&)>& function) const {
    auto dim = size();
    size_t rows = std::get<0>(dim);
    size_t cols = std::get<1>(dim);
    Matrix<ReturnType> result(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
      for (size_t j = 0; j < cols; ++j) {
        result.at(i, j) = function(at(i, j));
      }
    }
    return result;
  }

  string to_string() const {
    auto dim = size();
    size_t rows = std::get<0>(dim);
    size_t cols = std::get<1>(dim);

    std::stringstream out;
    out << "{\n";
    for (size_t i = 0; i < rows; ++i) {
      out << "{";
      for (size_t j = 0; j < cols; ++j) {
        out << at(i, j);
        if (j != cols - 1) {
          out << ", ";
        }
      }
      out << "}\n";
    }
    out << "}\n";
    return out.str();
  }

 private:
  // Empty curly braces means default value-initialize to zeros.
  std::vector<std::vector<T>> data_{};
};

#endif /* DYNAMIC_MATRIX_H */

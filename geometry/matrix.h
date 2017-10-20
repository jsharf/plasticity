#ifndef MATRIX_H
#define MATRIX_H

#include <array>
#include <functional>
#include <iostream>
#include <string>

using std::string;

template <size_t ROWS, size_t COLS, typename T>
class Matrix {
  using SimularMatrix = Matrix<ROWS, COLS, T>;

 public:
  Matrix(std::initializer_list<std::initializer_list<T>> values) {
    size_t i = 0;
    size_t j = 0;
    for (auto list : values) {
      j = 0;
      for (auto value : list) {
        at(i, j) = value;
        j++;
      }
      i++;
    }
  }
  Matrix() {}

  explicit Matrix(T value) {
    for (size_t i = 0; i < ROWS; ++i) {
      for (size_t j = 0; j < COLS; ++j) {
        at(i, j) = value;
      }
    }
  }

  static constexpr Matrix<ROWS, COLS, T>& Eye() {
    SimularMatrix result;
    for (size_t i = 0; i < ROWS; ++i) {
      result.at(i, i) = 1;
    }
    return result;
  }

  constexpr Matrix<ROWS, COLS, T>& operator*(T n) {
    SimularMatrix res;
    for (size_t i = 0; i < ROWS; ++i) {
      for (size_t j = 0; j < COLS; ++j) {
        res.at(i, j) = at(i, j) * n;
      }
    }
    return res;
  }

  constexpr Matrix<ROWS, COLS, T>& operator+(Matrix<ROWS, COLS, T> rhs) {
    SimularMatrix res;
    for (size_t i = 0; i < ROWS; ++i) {
      for (size_t j = 0; j < COLS; ++j) {
        res.at(i, j) = at(i, j) + rhs.at(i, j);
      }
    }
    return res;
  }

  constexpr const T& at(size_t i, size_t j) const { return data_.at(i).at(j); }
  constexpr T& at(size_t i, size_t j) { return data_.at(i).at(j); }

  // Return value is a std::pair of Matrices <lower, upper>.
  constexpr std::pair<Matrix<ROWS, COLS, T>, Matrix<ROWS, COLS, T>> LUDecomp() {
    if (ROWS != COLS) {
      std::cerr << "Warning: LUDecomp requested for non-square matrix."
                << std::endl;
    }
    SimularMatrix lower, upper;

    for (size_t i = 0; i < ROWS; ++i) {
      upper.at(i, i) = 1;
    }

    for (size_t j = 0; j < ROWS; ++j) {
      for (size_t i = j; i < ROWS; ++i) {
        T sum = 0;
        for (size_t k = 0; k < j; ++k) {
          sum = sum + lower.at(i, k) * upper.at(k, j);
        }
        lower.at(i, j) = at(i, j) - sum;
      }

      for (size_t i = j; i < ROWS; ++i) {
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

  constexpr Matrix<ROWS, 1, T> LUSolve(Matrix<ROWS, 1, T> b) {
    Matrix<ROWS, 1, T> d, x;
    auto matpair = LUDecomp();
    auto& lower = matpair.first;
    auto& upper = matpair.second;
    d.at(0, 0) = b.at(0, 0) / lower.at(0, 0);
    for (size_t i = 1; i < ROWS; ++i) {
      T sum = 0;
      for (size_t j = 0; j < i; ++j) {
        sum += lower.at(i, j) * d.at(j, 0);
      }
      d.at(i, 0) = (b.at(i, 0) - sum) / lower.at(i, i);
    }

    x.at(ROWS - 1, 0) = d.at(ROWS - 1, 0);
    for (int i = ROWS - 2; i >= 0; --i) {
      T sum = 0;
      for (size_t j = i + 1; j < ROWS; ++j) {
        sum += upper.at(i, j) * x.at(j, 0);
      }
      x.at(i, 0) = d.at(i, 0) - sum;
    }

    return x;
  }

  constexpr Matrix<ROWS, COLS, T> Invert() {
    using Colvec = Matrix<ROWS, 1, T>;
    std::array<Colvec, COLS> columns{};
    for (size_t i = 0; i < COLS; ++i) {
      columns[i].at(i, 0) = 1;
      columns[i] = LUSolve(columns[i]);
    }

    Matrix<ROWS, COLS, T> result;
    for (size_t i = 0; i < ROWS; ++i) {
      for (size_t j = 0; j < COLS; ++j) {
        result.at(i, j) = columns[j].at(i, 0);
      }
    }
    return result;
  }

  // Matrix multiplication is only possible if # COLS of LHS = # ROWS of RHS
  template <size_t RHSCOLS>
  Matrix<ROWS, RHSCOLS, T> operator*(Matrix<COLS, RHSCOLS, T> rhs) {
    Matrix<ROWS, RHSCOLS, T> result;
    for (size_t i = 0; i < ROWS; ++i) {
      for (size_t j = 0; j < RHSCOLS; ++j) {
        T kSum = 0;
        for (size_t k = 0; k < COLS; ++k) {
          kSum += at(i, k) * rhs.at(k, j);
        }
        result.at(i, j) = kSum;
      }
    }
    return result;
  }

  Matrix<COLS, ROWS, T> transpose() {
    Matrix<COLS, ROWS, T> result;
    for (size_t i = 0; i < ROWS; ++i) {
      for (size_t j = 0; j < COLS; ++j) {
        result.at(j, i) = at(i, j);
      }
    }
    return result;
  }

  template <typename ReturnType>
  Matrix<ROWS, COLS, ReturnType> map(
      const std::function<ReturnType(const T&)>& function) {
    Matrix<ROWS, COLS, ReturnType> result;
    for (size_t i = 0; i < ROWS; ++i) {
      for (size_t j = 0; j < COLS; ++j) {
        result.at(i, j) = function(at(i, j));
      }
    }
    return result;
  }

  string to_string() const {
    string result = "{\n";
    for (size_t i = 0; i < ROWS; ++i) {
      string row = "{";
      for (size_t j = 0; j < COLS; ++j) {
        row += std::to_string(at(i, j));
        if (j != COLS - 1) {
          row += ", ";
        }
      }
      row += "}";
      result += row + "\n";
    }
    result += "}\n";
    return result;
  }

 private:
  // Empty curly braces means default value-initialize to zeros.
  std::array<std::array<T, COLS>, ROWS> data_{};
};

#endif /* MATRIX_H */

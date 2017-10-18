#include "matrix4.h"

#include <cmath>
#include <string>
#include <utility>

// Matrix4 Operator Overloads

// Mat4 = N * Mat4
Matrix4 operator*(Number lhs, const Matrix4& rhs) {
  Matrix4 ret;
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      ret[i][j] = lhs * rhs[i][j];
    }
  }
  return ret;
}

// Mat4 = Mat4 * N
Matrix4 operator*(const Matrix4& lhs, Number rhs) {
  Matrix4 ret;
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      ret[i][j] = rhs * lhs[i][j];
    }
  }
  return ret;
}

// Mat4 = Mat4 * Mat4
Matrix4 operator*(const Matrix4& lhs, const Matrix4& rhs) {
  Matrix4 A;
  A[0][0] = lhs[0][0] * rhs[0][0] + lhs[0][1] * rhs[1][0] +
            lhs[0][2] * rhs[2][0] + lhs[0][3] * rhs[3][0];
  A[0][1] = lhs[0][0] * rhs[0][1] + lhs[0][1] * rhs[1][1] +
            lhs[0][2] * rhs[2][1] + lhs[0][3] * rhs[3][1];
  A[0][2] = lhs[0][0] * rhs[0][2] + lhs[0][1] * rhs[1][2] +
            lhs[0][2] * rhs[2][2] + lhs[0][3] * rhs[3][2];
  A[0][3] = lhs[0][0] * rhs[0][3] + lhs[0][1] * rhs[1][3] +
            lhs[0][2] * rhs[2][3] + lhs[0][3] * rhs[3][3];
  A[1][0] = lhs[1][0] * rhs[0][0] + lhs[1][1] * rhs[1][0] +
            lhs[1][2] * rhs[2][0] + lhs[1][3] * rhs[3][0];
  A[1][1] = lhs[1][0] * rhs[0][1] + lhs[1][1] * rhs[1][1] +
            lhs[1][2] * rhs[2][1] + lhs[1][3] * rhs[3][1];
  A[1][2] = lhs[1][0] * rhs[0][2] + lhs[1][1] * rhs[1][2] +
            lhs[1][2] * rhs[2][2] + lhs[1][3] * rhs[3][2];
  A[1][3] = lhs[1][0] * rhs[0][3] + lhs[1][1] * rhs[1][3] +
            lhs[1][2] * rhs[2][3] + lhs[1][3] * rhs[3][3];
  A[2][0] = lhs[2][0] * rhs[0][0] + lhs[2][1] * rhs[1][0] +
            lhs[2][2] * rhs[2][0] + lhs[2][3] * rhs[3][0];
  A[2][1] = lhs[2][0] * rhs[0][1] + lhs[2][1] * rhs[1][1] +
            lhs[2][2] * rhs[2][1] + lhs[2][3] * rhs[3][1];
  A[2][2] = lhs[2][0] * rhs[0][2] + lhs[2][1] * rhs[1][2] +
            lhs[2][2] * rhs[2][2] + lhs[2][3] * rhs[3][2];
  A[2][3] = lhs[2][0] * rhs[0][3] + lhs[2][1] * rhs[1][3] +
            lhs[2][2] * rhs[2][3] + lhs[2][3] * rhs[3][3];
  A[3][0] = lhs[3][0] * rhs[0][0] + lhs[3][1] * rhs[1][0] +
            lhs[3][2] * rhs[2][0] + lhs[3][3] * rhs[3][0];
  A[3][1] = lhs[3][0] * rhs[0][1] + lhs[3][1] * rhs[1][1] +
            lhs[3][2] * rhs[2][1] + lhs[3][3] * rhs[3][1];
  A[3][2] = lhs[3][0] * rhs[0][2] + lhs[3][1] * rhs[1][2] +
            lhs[3][2] * rhs[2][2] + lhs[3][3] * rhs[3][2];
  A[3][3] = lhs[3][0] * rhs[0][3] + lhs[3][1] * rhs[1][3] +
            lhs[3][2] * rhs[2][3] + lhs[3][3] * rhs[3][3];

  return A;
}

// Vector4 = Mat4 * Vector4
Vector4 operator*(const Matrix4& lhs, const Vector4& rhs) {
  Vector4 ret;

  ret.i = rhs.i * lhs[0][0] + rhs.j * lhs[0][1] + rhs.k * lhs[0][2] +
          rhs.r * lhs[0][3];
  ret.j = rhs.i * lhs[1][0] + rhs.j * lhs[1][1] + rhs.k * lhs[1][2] +
          rhs.r * lhs[1][3];
  ret.k = rhs.i * lhs[2][0] + rhs.j * lhs[2][1] + rhs.k * lhs[2][2] +
          rhs.r * lhs[2][3];
  ret.r = rhs.i * lhs[3][0] + rhs.j * lhs[3][1] + rhs.k * lhs[3][2] +
          rhs.r * lhs[3][3];

  return ret;
}

Vector3 operator*(const Matrix4& lhs, const Vector3& rhs) {
  Vector4 rhs_4d;
  rhs_4d.i = rhs.i;
  rhs_4d.j = rhs.j;
  rhs_4d.k = rhs.k;
  rhs_4d.r = 1;
  Vector4 result = lhs * rhs_4d;
  Vector3 result_3d;
  result_3d.i = result.i;
  result_3d.j = result.j;
  result_3d.k = result.k;
  return result_3d;
}

// Method implementations for Matrix4
Matrix4::Matrix4() {
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) (*this)[i][j] = 0;
}

Matrix4::Matrix4(Number other[4][4]) {
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) (*this)[i][j] = other[i][j];
}

Matrix4::Matrix4(const Matrix4& other) {
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) (*this)[i][j] = other[i][j];
}

Matrix4 Matrix4::eye() {
  Matrix4 A;
  for (int i = 0; i < 4; i++) A[i][i] = 1;
  return std::move(A);
}

Matrix4 Matrix4::Rot(const Vector3& a, Number theta) {
  Number c = cos(theta);
  Number s = sin(theta);

  Matrix4 A;

  A[0][0] = c + pow(a.i, 2) * (1 - c);
  A[0][1] = a.i * a.j * (1 - c) - a.k * s;
  A[0][2] = a.i * a.k * (1 - c) + a.j * s;
  A[1][0] = a.j * a.i * (1 - c) + a.k * s;
  A[1][1] = c + pow(a.j, 2) * (1 - c);
  A[1][2] = a.j * a.k * (1 - c) - a.i * s;
  A[2][0] = a.k * a.i * (1 - c) - a.j * s;
  A[2][1] = a.k * a.j * (1 - c) + a.i * s;
  A[2][2] = c + pow(a.k, 2) * (1 - c);

  A[0][3] = A[1][3] = A[2][3] = A[3][0] = A[3][1] = A[3][2] = 0;

  A[3][3] = 1;

  return A;
}

Matrix4 Matrix4::Rot(const Quaternion& q) {
  Number n = q.r * q.r + q.i * q.i + q.j * q.j + q.k * q.k;
  Number s = (n == 0) ? 0 : 2.0 / n;

  Number ri = s * q.r * q.i;
  Number rj = s * q.r * q.j;
  Number rk = s * q.r * q.k;

  Number ii = s * q.i * q.i;
  Number ij = s * q.i * q.j;
  Number ik = s * q.i * q.k;

  Number jj = s * q.j * q.j;
  Number jk = s * q.j * q.k;
  Number kk = s * q.k * q.k;

  Number a[4][4] = {{1 - (jj + kk), ij - rk, ik + rj, 0},
                    {ij + rk, 1 - (ii + kk), jk - ri, 0},
                    {ik - rj, jk + ri, 1 - (ii + jj), 0},
                    {0, 0, 0, 1}};

  return Matrix4(a);
}

Matrix4 Matrix4::RotI(Number theta) {
  Number c = cos(theta);
  Number s = sin(theta);

  Number ret[4][4] = {{1, 0, 0, 0}, {0, c, -s, 0}, {0, s, c, 0}, {0, 0, 0, 1}};

  return Matrix4(ret);
}

Matrix4 Matrix4::RotJ(Number theta) {
  Number c = cos(theta);
  Number s = sin(theta);

  Number ret[4][4] = {{c, 0, s, 0}, {0, 1, 0, 0}, {-s, 0, c, 0}, {0, 0, 0, 1}};

  return Matrix4(ret);
}

Matrix4 Matrix4::RotK(Number theta) {
  Number c = cos(theta);
  Number s = sin(theta);

  Number ret[4][4] = {{c, -s, 0, 0}, {s, c, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}};
  return Matrix4(ret);
}

Matrix4 Matrix4::Translate(Number i, Number j, Number k) {
  Number ret[4][4] = {{1, 0, 0, i}, {0, 1, 0, j}, {0, 0, 1, k}, {0, 0, 0, 1}};
  return Matrix4(ret);
}

Matrix4 Matrix4::Scale(Number i, Number j, Number k) {
  Number ret[4][4] = {{i, 0, 0, 0}, {0, j, 0, 0}, {0, 0, k, 0}, {0, 0, 0, 1}};
  return Matrix4(ret);
}

Matrix4& Matrix4::operator*=(Number n) {
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++) data_[i][j] *= n;

  return *this;
}

Matrix4& Matrix4::operator+=(const Matrix4& rhs) {
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++) data_[i][j] += rhs[i][j];

  return *this;
}

Matrix4 Matrix4::Transpose() const {
  Matrix4 ret;

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      ret[j][i] = data_[i][j];
    }
  }

  return ret;
}

Matrix4 Matrix4::Invert() const {
  //
  // Inversion by Cramer's rule.  Code taken from an Intel publication
  //
  Number result[4][4];
  Number tmp[12]; /* temp array for pairs */
  Number src[16]; /* array of transpose source matrix */
  Number det;     /* determinant */
  /* transpose matrix */
  for (int i = 0; i < 4; ++i) {
    src[i + 0] = data_[i][0];
    src[i + 4] = data_[i][1];
    src[i + 8] = data_[i][2];
    src[i + 12] = data_[i][3];
  }
  /* calculate pairs for first 8 elements (cofactors) */
  tmp[0] = src[10] * src[15];
  tmp[1] = src[11] * src[14];
  tmp[2] = src[9] * src[15];
  tmp[3] = src[11] * src[13];
  tmp[4] = src[9] * src[14];
  tmp[5] = src[10] * src[13];
  tmp[6] = src[8] * src[15];
  tmp[7] = src[11] * src[12];
  tmp[8] = src[8] * src[14];
  tmp[9] = src[10] * src[12];
  tmp[10] = src[8] * src[13];
  tmp[11] = src[9] * src[12];
  /* calculate first 8 elements (cofactors) */
  result[0][0] = tmp[0] * src[5] + tmp[3] * src[6] + tmp[4] * src[7];
  result[0][0] -= tmp[1] * src[5] + tmp[2] * src[6] + tmp[5] * src[7];
  result[0][1] = tmp[1] * src[4] + tmp[6] * src[6] + tmp[9] * src[7];
  result[0][1] -= tmp[0] * src[4] + tmp[7] * src[6] + tmp[8] * src[7];
  result[0][2] = tmp[2] * src[4] + tmp[7] * src[5] + tmp[10] * src[7];
  result[0][2] -= tmp[3] * src[4] + tmp[6] * src[5] + tmp[11] * src[7];
  result[0][3] = tmp[5] * src[4] + tmp[8] * src[5] + tmp[11] * src[6];
  result[0][3] -= tmp[4] * src[4] + tmp[9] * src[5] + tmp[10] * src[6];
  result[1][0] = tmp[1] * src[1] + tmp[2] * src[2] + tmp[5] * src[3];
  result[1][0] -= tmp[0] * src[1] + tmp[3] * src[2] + tmp[4] * src[3];
  result[1][1] = tmp[0] * src[0] + tmp[7] * src[2] + tmp[8] * src[3];
  result[1][1] -= tmp[1] * src[0] + tmp[6] * src[2] + tmp[9] * src[3];
  result[1][2] = tmp[3] * src[0] + tmp[6] * src[1] + tmp[11] * src[3];
  result[1][2] -= tmp[2] * src[0] + tmp[7] * src[1] + tmp[10] * src[3];
  result[1][3] = tmp[4] * src[0] + tmp[9] * src[1] + tmp[10] * src[2];
  result[1][3] -= tmp[5] * src[0] + tmp[8] * src[1] + tmp[11] * src[2];
  /* calculate pairs for second 8 elements (cofactors) */
  tmp[0] = src[2] * src[7];
  tmp[1] = src[3] * src[6];
  tmp[2] = src[1] * src[7];
  tmp[3] = src[3] * src[5];
  tmp[4] = src[1] * src[6];
  tmp[5] = src[2] * src[5];

  tmp[6] = src[0] * src[7];
  tmp[7] = src[3] * src[4];
  tmp[8] = src[0] * src[6];
  tmp[9] = src[2] * src[4];
  tmp[10] = src[0] * src[5];
  tmp[11] = src[1] * src[4];
  /* calculate second 8 elements (cofactors) */
  result[2][0] = tmp[0] * src[13] + tmp[3] * src[14] + tmp[4] * src[15];
  result[2][0] -= tmp[1] * src[13] + tmp[2] * src[14] + tmp[5] * src[15];
  result[2][1] = tmp[1] * src[12] + tmp[6] * src[14] + tmp[9] * src[15];
  result[2][1] -= tmp[0] * src[12] + tmp[7] * src[14] + tmp[8] * src[15];
  result[2][2] = tmp[2] * src[12] + tmp[7] * src[13] + tmp[10] * src[15];
  result[2][2] -= tmp[3] * src[12] + tmp[6] * src[13] + tmp[11] * src[15];
  result[2][3] = tmp[5] * src[12] + tmp[8] * src[13] + tmp[11] * src[14];
  result[2][3] -= tmp[4] * src[12] + tmp[9] * src[13] + tmp[10] * src[14];
  result[3][0] = tmp[2] * src[10] + tmp[5] * src[11] + tmp[1] * src[9];
  result[3][0] -= tmp[4] * src[11] + tmp[0] * src[9] + tmp[3] * src[10];
  result[3][1] = tmp[8] * src[11] + tmp[0] * src[8] + tmp[7] * src[10];
  result[3][1] -= tmp[6] * src[10] + tmp[9] * src[11] + tmp[1] * src[8];
  result[3][2] = tmp[6] * src[9] + tmp[11] * src[11] + tmp[3] * src[8];
  result[3][2] -= tmp[10] * src[11] + tmp[2] * src[8] + tmp[7] * src[9];
  result[3][3] = tmp[10] * src[10] + tmp[4] * src[8] + tmp[9] * src[9];
  result[3][3] -= tmp[8] * src[9] + tmp[11] * src[10] + tmp[5] * src[8];
  /* calculate determinant */
  det = src[0] * result[0][0] + src[1] * result[0][1] + src[2] * result[0][2] +
        src[3] * result[0][3];
  if (det == 0) {
    return *this;
  }
  /* calculate matrix inverse */
  Number determinant_multiplier = 1.0f / det;

  Matrix4 normalized_result;
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      normalized_result[i][j] =
          static_cast<Number>(result[i][j] * determinant_multiplier);
    }
  }
  return normalized_result;
}

Number* Matrix4::operator[](int i) const { return ((Number*)data_[i]); }

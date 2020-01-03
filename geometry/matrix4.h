#ifndef Matrix4_H
#define Matrix4_H

#include "plasticity/geometry/types.h"
#include "plasticity/geometry/vector.h"

class Matrix4 {
 public:
  Matrix4();
  Matrix4(Number other[4][4]);
  Matrix4(const Matrix4& other);
  static Matrix4 eye();
  static Matrix4 Rot(const Vector3& a, Number theta);
  static Matrix4 Rot(const Quaternion& q);
  static Matrix4 RotI(Number theta);
  static Matrix4 RotJ(Number theta);
  static Matrix4 RotK(Number theta);
  static Matrix4 Translate(Number i, Number j, Number k);
  static Matrix4 Scale(Number i, Number j, Number k);
  Matrix4& operator+=(const Matrix4& rhs);
  Matrix4& operator*=(Number n);
  Matrix4 Transpose() const;
  Matrix4 Invert() const;
  Number *operator[](int i) const;

 private:
  Number data_[4][4];
};

// Matrix4 Operator Overloads Declarations
Matrix4 operator*(Number lhs, const Matrix4& rhs);
Matrix4 operator*(const Matrix4& lhs, Number rhs);
Matrix4 operator*(const Matrix4& lhs, const Matrix4& rhs);
Vector4 operator*(const Matrix4& lhs, const Vector4& rhs);
Vector3 operator*(const Matrix4& lhs, const Vector3& rhs);

#endif /* Matrix4_H */

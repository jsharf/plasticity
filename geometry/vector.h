#ifndef VECTOR_H
#define VECTOR_H

#include <cmath>

#include "types.h"

struct Vector2 {
  Vector2() : i(0), j(0) {}
  Vector2(Number i0, Number j0) : i(i0), j(j0) {}
  Number i, j;
};

struct Vector3 {
  Vector3() : i(0), j(0), k(0) {}
  Vector3(Number i0, Number j0, Number k0) : i(i0), j(j0), k(k0) {}
  Number Magnitude() const { return sqrt(pow(i, 2) + pow(j, 2) + pow(k, 2)); }
  Vector3 Normalize() const {
    const Number mag = Magnitude();
    return Vector3(i / mag, j / mag, k / mag);
  }
  Number i, j, k;
};

struct Vector4 {
  Vector4() : i(0), j(0), k(0), r(0) {}
  Vector4(Number i0, Number j0, Number k0, Number r0)
      : i(i0), j(j0), k(k0), r(r0) {}
  Number i, j, k, r;
};

// Quaternion and Vector4 have same state-space but different behaviors
struct Quaternion : public Vector4 {
  Quaternion() : Vector4(0, 0, 0, 1) {}
  Quaternion(const Vector3& axis, const Number angle) {
    Vector3 norm_axis = axis.Normalize();
    i = norm_axis.i * sin(angle / 2.0);
    j = norm_axis.j * sin(angle / 2.0);
    k = norm_axis.k * sin(angle / 2.0);
    r = cos(angle / 2.0);
  }
  Vector3 getAxis() const;
  Number getAngle() const;
};

Vector4 operator*(Number lhs, const Vector4& rhs);
Vector4 operator*(const Vector4& rhs, Number lhs);
Vector3 operator*(Number lhs, const Vector3& rhs);
Vector3 operator*(const Vector3& rhs, Number lhs);
Vector2 operator*(Number lhs, const Vector2& rhs);
Vector2 operator*(const Vector2& rhs, Number lhs);

#endif /* VECTOR_H */

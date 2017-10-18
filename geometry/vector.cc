#include "vector.h"

#include <cmath>

Vector3 Quaternion::getAxis() const
{

    Number theta = getAngle();
    
    Vector3 a;

    Number denominator = sin(0.5*theta);

    a.i = i/denominator;
    a.j = j/denominator;
    a.k = k/denominator;
    return a;
}

Number Quaternion::getAngle() const
{
    return 2.0*acos(r);
}

// Vector4 = N * Vector4
Vector4 operator*(Number lhs, const Vector4& rhs)
{
    Vector4 ret;
    ret.i = lhs * rhs.i;
    ret.j = lhs * rhs.j;
    ret.k = lhs * rhs.k;
    ret.r = lhs * rhs.r;
    return ret;
}

// Vector4 = Vector4 * N
Vector4 operator*(const Vector4& rhs, Number lhs)
{
    Vector4 ret;
    ret.i = lhs * rhs.i;
    ret.j = lhs * rhs.j;
    ret.k = lhs * rhs.k;
    ret.r = lhs * rhs.r;
    return ret;
}

// Vector3 = N * Vector3
Vector3 operator*(Number lhs, const Vector3& rhs)
{
    Vector3 ret;
    ret.i = lhs * rhs.i;
    ret.j = lhs * rhs.j;
    ret.k = lhs * rhs.k;
    return ret;
}

// Vector3 = Vector3 * N
Vector3 operator*(const Vector3& rhs, Number lhs)
{
    Vector3 ret;
    ret.i = lhs * rhs.i;
    ret.j = lhs * rhs.j;
    ret.k = lhs * rhs.k;
    return ret;
}

// Vector2 = N * Vector4
Vector2 operator*(Number lhs, const Vector2& rhs)
{
    Vector2 ret;
    ret.i = lhs * rhs.i;
    ret.j = lhs * rhs.j;
    return ret;
}

// Vector2 = Vector2 * N
Vector2 operator*(const Vector2& rhs, Number lhs)
{
    Vector2 ret;
    ret.i = lhs * rhs.i;
    ret.j = lhs * rhs.j;
    return ret;
}

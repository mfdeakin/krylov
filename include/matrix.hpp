
#ifndef _MATRIX_HPP_
#define _MATRIX_HPP_

#include <cmath>

// Just matrix multiplication implementations
matrix &dot(matrix &result, const matrix &lhs, const matrix &rhs) noexcept {
  assert(lhs.shape()[1] == rhs.shape()[0]);
  assert(result.shape()[0] == lhs.shape()[0]);
  assert(result.shape()[1] == rhs.shape()[1]);
  for (unsigned long i = 0; i < lhs.shape()[0]; i++) {
    for (unsigned long j = 0; j < rhs.shape()[1]; j++) {
      result(i, j) = 0.0;
      for (unsigned long k = 0; k < lhs.shape()[1]; k++) {
        result(i, j) += lhs(i, k) * rhs(k, j);
      }
    }
  }
  return result;
}

vector &dot(vector &result, const matrix &lhs, const vector &rhs) noexcept {
  assert(lhs.shape()[1] == rhs.shape()[0]);
  assert(result.shape()[0] == lhs.shape()[0]);
  for (unsigned long i = 0; i < lhs.shape()[0]; i++) {
    result(i) = 0.0;
    for (unsigned long k = 0; k < lhs.shape()[1]; k++) {
      result(i) += lhs(i, k) * rhs(k);
    }
  }
  return result;
}

real dot(real &result, const vector &lhs, const vector &rhs) noexcept {
  // Allow the lhs vector to be shorter than the rhs to make use in GMRES easier
  // This treats all of the non-existent entries in lhs as 0
  assert(lhs.shape()[0] <= rhs.shape()[0]);
  result = 0.0;
  for (unsigned long i = 0; i < lhs.shape()[0]; i++) {
    result += lhs(i) * rhs(i);
  }
  return result;
}

real l2_norm(const vector &v) noexcept {
  real l2 = 0.0;
  for (real c : v) {
    l2 += c * c;
  }
  return std::sqrt(l2);
}

real normalize(vector &v) noexcept {
  real l2 = l2_norm(v);
  for (real &r : v) {
    r /= l2;
  }
  return l2;
}

#endif // _MATRIX_HPP_

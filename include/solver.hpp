
#ifndef _SOLVER_HPP_
#define _SOLVER_HPP_

#include "constants.hpp"

std::pair<matrix, matrix> lu_decomp(matrix system);

vector lu_solve(matrix system, vector sol) {
  const auto [l, u] = lu_decomp(system);
  vector partial(vector::shape_type({system.shape()[1]}));
  // Solve L y = b
  for (int i = 0; i < system.shape()[0]; i++) {
    partial(i) = sol(i);
    for (int j = 0; j < i; j++) {
      partial(i) -= l(i, j) * partial(j);
    }
    partial(i) /= l(i, i);
  }
  // Solve U x = y
  vector vars(vector::shape_type({system.shape()[1]}));
  for (int i = system.shape()[0] - 1; i >= 0; i--) {
    vars(i) = partial(i);
    for (int j = i + 1; j < system.shape()[0]; j++) {
      vars(i) -= u(i, j) * vars(j);
    }
    vars(i) /= u(i, i);
  }
  return vars;
}

std::pair<matrix, matrix> lu_decomp(matrix system) {
  // Only implemented for square matrices
  assert(system.shape()[0] == system.shape()[1]);
  xt::xtensor<real, 2> l(system.shape()), u(system.shape());
  for (real &v : l) {
    v = 0.0;
  }
  for (real &v : u) {
    v = 0.0;
  }
  // Use Doolittle Factorization
  for (int k = 0; k < system.shape()[0]; k++) {
    l(k, k) = 1.0;

    u(k, k) = system(k, k);
    for (int j = 0; j < k; j++) {
      u(k, k) -= l(k, j) * u(j, k);
    }
    u(k, k) /= l(k, k);

    for (int j = k + 1; j < system.shape()[0]; j++) {
      u(k, j) = system(k, j);
      for (int i = 0; i < k; i++) {
        u(k, j) -= l(k, i) * u(i, j);
      }
      u(k, j) /= l(k, k);
    }

    for (int j = k + 1; j < system.shape()[0]; j++) {
      l(j, k) = system(j, k);
      for (int i = 0; i < k; i++) {
        l(j, k) -= l(j, i) * u(i, k);
      }
      l(j, k) /= u(k, k);
    }
  }
  return {l, u};
}

matrix dot(matrix lhs, matrix rhs) {
  assert(lhs.shape()[1] == rhs.shape()[0]);
  matrix result({lhs.shape()[0], rhs.shape()[1]});
  for (int i = 0; i < lhs.shape()[0]; i++) {
    for (int j = 0; j < rhs.shape()[1]; j++) {
      result(i, j) = 0.0;
      for (int k = 0; k < lhs.shape()[1]; k++) {
        result(i, j) += lhs(i, k) * rhs(k, j);
      }
    }
  }
  return result;
}

vector dot(matrix lhs, vector rhs) {
  assert(lhs.shape()[1] == rhs.shape()[0]);
  vector result(vector::shape_type({lhs.shape()[1]}));
  for (int i = 0; i < lhs.shape()[0]; i++) {
    result(i) = 0.0;
    for (int k = 0; k < lhs.shape()[1]; k++) {
      result(i) += lhs(i, k) * rhs(k);
    }
  }
  return result;
}

#endif // _SOLVER_HPP_

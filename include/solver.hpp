
#ifndef _SOLVER_HPP_
#define _SOLVER_HPP_

#include "constants.hpp"

std::pair<matrix, matrix> lu_decomp(matrix system);

template <typename Impl_> class Solver {
public:
  using Impl = Impl_;

  explicit Solver(unsigned long sys_size)
      : vars(vector::shape_type({sys_size})) {
    for (int i = 0; i < sys_size; i++) {
      vars(i) = s_nan;
    }
  }

  auto update(matrix system, vector sol) noexcept {
    Impl::solve(system, sol);
    return xt::view(vars);
  }

protected:
  vector vars;
};

// LU Solver for positive definite matrices, no partial pivoting
class LUSolver : Solver<LUSolver> {
public:
  explicit LUSolver(unsigned long sys_size)
      : Solver<LUSolver>(sys_size), l({sys_size, sys_size}),
        u({sys_size, sys_size}), partial(vector::shape_type{sys_size}) {}

  vector solve(matrix system, vector sol) {
    assert(system.shape()[0] == l.shape()[0]);
    assert(system.shape()[1] == l.shape()[1]);
    // l and u must be computed for the system
    lu_decomp(system);
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

  std::pair<const matrix &, const matrix &> lu_decomp(matrix system) {
    // Construct lower (l) and upper (u) matrices s.t. system = l * u
    // Only implemented for square matrices
    assert(system.shape()[0] == system.shape()[1]);
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

protected:
  // better would be to use a single matrix, as the indeterminant entries are
  // the complement in each matrix
  // 2 matrices is easier to understand however
  matrix l, u;
  vector partial;
};

#endif // _SOLVER_HPP_

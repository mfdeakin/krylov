
#ifndef _SOLVER_HPP_
#define _SOLVER_HPP_

#include <memory>

#include "constants.hpp"
#include "matrix.hpp"

std::pair<matrix, matrix> lu_decomp(matrix system);

class Solver {
public:
  explicit Solver(unsigned long sys_size)
      : vars(vector::shape_type({sys_size})) {
    for (unsigned long i = 0; i < sys_size; i++) {
      vars(i) = s_nan;
    }
  }

  const vector &prev_result() const noexcept { return vars; }

protected:
  vector vars;
};

// LU Solver for positive definite matrices, no partial pivoting
class LUSolver : public Solver {
public:
  explicit LUSolver(unsigned long sys_size)
      : Solver(sys_size), l({sys_size, sys_size}), u({sys_size, sys_size}),
        partial(vector::shape_type{sys_size}) {}

  const vector &solve(const matrix &system, const vector &sol) {
    assert(system.shape()[0] == l.shape()[0]);
    assert(system.shape()[1] == l.shape()[1]);
    // l and u must be computed for the system
    lu_decomp(system);
    // Solve L y = b
    for (unsigned long i = 0; i < system.shape()[0]; i++) {
      partial(i) = sol(i);
      for (unsigned long j = 0; j < i; j++) {
        partial(i) -= l(i, j) * partial(j);
      }
      partial(i) /= l(i, i);
    }
    // Solve U x = y
    for (unsigned long i = system.shape()[0] - 1; i < system.shape()[0]; i--) {
      vars(i) = partial(i);
      for (unsigned long j = i + 1; j < system.shape()[0]; j++) {
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
    for (unsigned long k = 0; k < system.shape()[0]; k++) {
      l(k, k) = 1.0;

      u(k, k) = system(k, k);
      for (unsigned long j = 0; j < k; j++) {
        u(k, k) -= l(k, j) * u(j, k);
      }
      u(k, k) /= l(k, k);
      assert(u(k, k) != 0.0);

      for (unsigned long j = k + 1; j < system.shape()[0]; j++) {
        u(k, j) = system(k, j);
        for (unsigned long i = 0; i < k; i++) {
          u(k, j) -= l(k, i) * u(i, j);
        }
        u(k, j) /= l(k, k);
      }

      for (unsigned long j = k + 1; j < system.shape()[0]; j++) {
        l(j, k) = system(j, k);
        for (unsigned long i = 0; i < k; i++) {
          l(j, k) -= l(j, i) * u(i, k);
        }
        l(j, k) /= u(k, k);
      }
    }
    return {l, u};
  }

  unsigned long resize(const unsigned long new_size) {
    l = matrix(matrix::shape_type{new_size, new_size});
    u = matrix(matrix::shape_type{new_size, new_size});
    partial = vector(vector::shape_type{new_size});
    return new_size;
  }

  unsigned long size() { return l.shape()[0]; }

protected:
  // better would be to use a single matrix, as the indeterminant entries are
  // the complement in each matrix
  // 2 matrices is easier to understand however
  matrix l, u;
  vector partial;
};

template <unsigned int num_iters> class GMRESSolver : public Solver {
public:
  explicit GMRESSolver(unsigned long sys_size)
      : Solver(sys_size), subspace(),
        H(matrix::shape_type{num_iters + 1, num_iters}),
        reduced_solver(num_iters) {}

  const vector &solve(const matrix &system, const vector &sol) {
    const real beta = initial_subspace(sol);
    for (unsigned long i = 0; i < H.shape()[0]; i++) {
      for (unsigned long j = 0; j < H.shape()[1]; j++) {
        H(i, j) = 0.0;
      }
    }
    // Presumably we only rarely want to use less iterations than previously
    for (unsigned long i = 0; i < reduced_solver.size(); i++) {
      const real mag = add_arnoldi(system);
      // If we have the zero vector, we have an exact solution in our subspace
      // This is unfortunate, as I don't handle this case yet
      assert(std::abs(mag) > 1e-20);
      // if (std::abs(mag) < 1e-10) {
      //   shrink_subspace();
      // }
    }

    // We have our subspace, now minimize $||\beta e_1 - H y||$
    // Using least squares, we solve \beta $H^T e_1 = H^T H y$ for y
    matrix sys(matrix::shape_type{H.shape()[1], H.shape()[1]});
    dot(sys, xt::transpose(H), H);
    vector lst_sq_sol(vector::shape_type{H.shape()[1]});
    for (unsigned long i = 0; i < H.shape()[1]; i++) {
      lst_sq_sol(i) = beta * H(0, i);
    }

    const vector &subsol = reduced_solver.solve(sys, lst_sq_sol);
    // We have our subspace solution! Now compute our actual solution
    for (unsigned long i = 0; i < vars.shape()[0]; i++) {
      vars(i) = 0.0;
    }
    for (unsigned long i = 0; i < subsol.shape()[0]; i++) {
      const vector &dim = subspace[i];
      for (unsigned long j = 0; j < vars.shape()[0]; j++) {
        vars(j) += subsol(i) * dim(j);
      }
    }
    return vars;
  }

  const vector &subspace_vector(int i) {
    assert(i < subspace.size());
    return subspace[i];
  }

  real initial_subspace(const vector &sol) {
    vector v1 = sol;
    const real beta = normalize(v1);
    subspace.clear();
    subspace.push_back(v1);
    return beta;
  }

  void shrink_subspace() {
    // Our subspace was larger than we needed, so shrink to fit
    reduced_solver.resize(subspace.size() - 1);
    matrix new_H(matrix::shape_type{subspace.size(), subspace.size() - 1});
    for (unsigned long i = 0; i < new_H.shape()[0]; i++) {
      for (unsigned long j = 0; j < new_H.shape()[1]; j++) {
        new_H(i, j) = H(i, j);
      }
    }
    std::swap(H, new_H);
  }

  // The compiler should be able to inline this function, so not using static
  // polymorphism here shouldn't terribly hurt performance
  virtual real add_arnoldi(const matrix &system) {
    assert(subspace.size() > 0);
    vector next(vector::shape_type{vars.shape()[0]});
    dot(next, system, *(subspace.end() - 1));
    const unsigned long j = subspace.size() - 1;
    // Apply Gram-Schmidt to compute the orthogonal component
    gram_schmidt(next, subspace);
    subspace.push_back(next);
    return H(j + 1, j);
  }

  vector &gram_schmidt(vector &next, const std::vector<vector> &subspace) {
    const unsigned long j = subspace.size() - 1;
    for (unsigned long i = 0; i <= j; i++) {
      const vector &v = subspace[i];
      assert(i < H.shape()[0]);
      assert(j < H.shape()[1]);
      dot(H(i, j), next, v);
      // All of the vectors in our subspace are orthogonal to each other, so
      // hij doesn't change as we subtract components off Thus, to avoid
      // repeating work, store it here
      next -= H(i, j) * v;
    }

    H(j + 1, j) = normalize(next);
    return next;
  }

  // A nicer version which doesn't assume how many iterations we're doing
  // This unfortunately likely doesn't perform as well, as it continuously
  // allocates new vectors
  real add_arnoldi(std::vector<vector> &subspace, std::vector<vector> &H,
                   const matrix &system) {
    assert(subspace.size() > 0);
    vector next(vector::shape_type{vars.shape()[0]});
    dot(next, system, *(subspace.end() - 1));
    vector hj(vector::shape_type{subspace.size() + 1});
    for (unsigned long i = 0; i < subspace.size(); i++) {
      const vector &v = subspace[i];
      dot(hj(i), next, v);
      for (unsigned long k = 0; k < vars.shape()[0]; k++) {
        next(k) -= hj(i) * v(k);
      }
    }
    hj(hj.shape()[0] - 1) = normalize(next);
    subspace.push_back(next);
    H.push_back(hj);
    return hj(hj.shape()[0] - 1);
  }

protected:
  std::vector<vector> subspace;
  matrix H;
  LUSolver reduced_solver;
};

class SparseTriDiagSolver : public Solver {
public:
  explicit SparseTriDiagSolver(unsigned long sys_size)
      : Solver(sys_size), u_diag(vector::shape_type{sys_size - 1}) {}

  const vector &solve(const matrix &diags, const vector &sol) {
    // diags(i, 0) is the lower diagonal entry on row i, if this entry exists
    // diags(i, 1) is the diagonal entry on row i, it always exists
    // diags(i, 2) is the upper diagonal entry on row i, if this entry exists
    assert(diags(0, 0) != 0.0);
    assert(diags.shape()[0] == u_diag.shape()[0] + 1);
    assert(diags.shape()[1] == 3);
    assert(sol.shape()[0] == u_diag.shape()[0] + 1);
    vars(0) = sol(0) / diags(0, 1);
    u_diag(0) = diags(0, 2) / diags(0, 1);
    // Forward substitution to clear the lower diagonal
    for (unsigned long i = 1; i < vars.shape()[0]; i++) {
      // We're subtracting diags(i, i - 1) * prev_row from this row
      // Then normalizing so that the diagonal of this row is 1.0
      const real diag = diags(i, 1) - u_diag(i - 1) * diags(i, 0);
      assert(diag != 0.0);
      vars(i) = (sol(i) - vars(i - 1) * diags(i, 0)) / diag;
      if (i + 1 < vars.shape()[0]) {
        // Normalize the upper diagonal as well
        u_diag(i) = diags(i, 2) / diag;
      }
    }
    // Now the back substitution
    for (unsigned long i = u_diag.shape()[0] - 1; i < u_diag.shape()[0]; i--) {
      vars(i) -= vars(i + 1) * u_diag(i);
    }
    return vars;
  }

protected:
  vector u_diag;
};

template <unsigned int num_iters>
class GMRESTDPSolver : public GMRESSolver<num_iters> {
public:
  using Parent = GMRESSolver<num_iters>;

  explicit GMRESTDPSolver(unsigned long sys_size)
      : Parent(sys_size), td(sys_size),
        sparse_tdiags(matrix::shape_type{sys_size, 3}),
        next_unp(vector::shape_type{sys_size}) {}

  const vector &solve(const matrix &system, const vector &sol) {
    // We need to initialize the sparse_tdiags matrix for preconditioning
    for (int i = 0; i < system.shape()[0]; i++) {
      sparse_tdiags(i, 0) = system(i, i - 1);
      sparse_tdiags(i, 1) = system(i, i);
      sparse_tdiags(i, 2) = system(i, i + 1);
    }
    Parent::solve(system, sol);
  }

  virtual real add_arnoldi(const matrix &system) {
    assert(this->subspace.size() > 0);
    dot(next_unp, system, *(this->subspace.end() - 1));
    vector next = td.solve(sparse_tdiags, next_unp);
    const unsigned long j = this->subspace.size() - 1;
    // Apply Gram-Schmidt to compute the orthogonal component
    gram_schmidt(next, this->subspace);
    this->subspace.push_back(next);
    return this->H(j + 1, j);
  }

protected:
  SparseTriDiagSolver td;
  matrix sparse_tdiags;
  vector next_unp;
};

class TriDiagSolver : public Solver {
public:
  explicit TriDiagSolver(unsigned long sys_size)
      : Solver(sys_size), u_diag(vector::shape_type{sys_size - 1}) {}

  const vector &solve(const matrix &system, const vector &sol) {
    assert(system(0, 0) != 0.0);
    assert(system.shape()[0] == u_diag.shape()[0] + 1);
    assert(system.shape()[1] == u_diag.shape()[0] + 1);
    assert(sol.shape()[0] == u_diag.shape()[0] + 1);
    vars(0) = sol(0) / system(0, 0);
    u_diag(0) = system(0, 1) / system(0, 0);
    // Forward substitution to clear the lower diagonal
    for (unsigned long i = 1; i < vars.shape()[0]; i++) {
      // We're subtracting system(i, i - 1) * prev_row from this row
      // Then normalizing so that the diagonal of this row is 1.0
      const real diag = system(i, i) - u_diag(i - 1) * system(i, i - 1);
      assert(diag != 0.0);
      vars(i) = (sol(i) - vars(i - 1) * system(i, i - 1)) / diag;
      if (i + 1 < vars.shape()[0]) {
        // Normalize the upper diagonal as well
        u_diag(i) = system(i, i + 1) / diag;
      }
    }
    // Now the back substitution
    for (unsigned long i = u_diag.shape()[0] - 1; i < u_diag.shape()[0]; i--) {
      vars(i) -= vars(i + 1) * u_diag(i);
    }
    return vars;
  }

protected:
  vector u_diag;
};

#endif // _SOLVER_HPP_

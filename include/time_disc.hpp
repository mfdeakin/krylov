
#ifndef _TIME_DISC_HPP_
#define _TIME_DISC_HPP_

#include "boundary_conds.hpp"
#include "mesh.hpp"
#include "solver.hpp"

#include "xtensor/xtensor.hpp"

using BCSpec = std::pair<BCType, std::function<real(real, real, real)>>;

template <typename SpaceDisc_> class TimeDisc {
public:
  using SpaceDisc = SpaceDisc_;

  TimeDisc(
      const std::pair<real, real> corner_1,
      const std::pair<real, real> corner_2, const size_t x_cells,
      const size_t y_cells, std::function<triple(real, real)> initial_t_u_v,
      const SpaceDisc &space_,
      // Assume all of the boundaries are homogeneous Dirichlet by default
      std::pair<BCType, std::function<real(real, real, real)>> bottom_temp_bc_ =
          {BCType::Dirichlet, [](real, real, real) { return 0.0; }},
      std::pair<BCType, std::function<real(real, real, real)>> top_temp_bc_ =
          {BCType::Dirichlet, [](real, real, real) { return 0.0; }},
      std::pair<BCType, std::function<real(real, real, real)>> left_temp_bc_ =
          {BCType::Dirichlet, [](real, real, real) { return 0.0; }},
      std::pair<BCType, std::function<real(real, real, real)>> right_temp_bc_ =
          {BCType::Dirichlet, [](real, real, real) { return 0.0; }},

      std::pair<BCType, std::function<real(real, real, real)>>
          bottom_vel_u_bc_ = {BCType::Dirichlet,
                              [](real, real, real) { return 0.0; }},
      std::pair<BCType, std::function<real(real, real, real)>> top_vel_u_bc_ =
          {BCType::Dirichlet, [](real, real, real) { return 0.0; }},
      std::pair<BCType, std::function<real(real, real, real)>> left_vel_u_bc_ =
          {BCType::Dirichlet, [](real, real, real) { return 0.0; }},
      std::pair<BCType, std::function<real(real, real, real)>> right_vel_u_bc_ =
          {BCType::Dirichlet, [](real, real, real) { return 0.0; }},

      std::pair<BCType, std::function<real(real, real, real)>>
          bottom_vel_v_bc_ = {BCType::Dirichlet,
                              [](real, real, real) { return 0.0; }},
      std::pair<BCType, std::function<real(real, real, real)>> top_vel_v_bc_ =
          {BCType::Dirichlet, [](real, real, real) { return 0.0; }},
      std::pair<BCType, std::function<real(real, real, real)>> left_vel_v_bc_ =
          {BCType::Dirichlet, [](real, real, real) { return 0.0; }},
      std::pair<BCType, std::function<real(real, real, real)>> right_vel_v_bc_ =
          {BCType::Dirichlet, [](real, real, real) { return 0.0; }})
      : cur_state(corner_1, corner_2, x_cells, y_cells, initial_t_u_v),
        space(space_), time(0.0),
        bottom_temp_bc(cur_state.ghostcells_bottom_Temp(),
                       cur_state.bndrycells_bottom_Temp(),
                       [=](const int i) { return cur_state.median_x(i); },
                       [=](const int i) { return cur_state.bottom_y(0); },
                       bottom_temp_bc_.second, cur_state.dy(),
                       bottom_temp_bc_.first),
        top_temp_bc(cur_state.ghostcells_top_Temp(),
                    cur_state.bndrycells_top_Temp(),
                    [=](const int i) { return cur_state.median_x(i); },
                    [=](const int i) {
                      return cur_state.top_y(cur_state.cells_y() - 1);
                    },
                    top_temp_bc_.second, cur_state.dy(), top_temp_bc_.first),
        left_temp_bc(cur_state.ghostcells_left_Temp(),
                     cur_state.bndrycells_right_Temp(),
                     [=](const int j) { return cur_state.left_x(0); },
                     [=](const int j) { return cur_state.median_y(j); },
                     left_temp_bc_.second, cur_state.dx(), left_temp_bc_.first),
        right_temp_bc(cur_state.ghostcells_right_Temp(),
                      cur_state.bndrycells_right_Temp(),
                      [=](const int j) {
                        return cur_state.right_x(cur_state.cells_x() - 1);
                      },
                      [=](const int j) { return cur_state.median_y(j); },
                      right_temp_bc_.second, cur_state.dx(),
                      right_temp_bc_.first),

        bottom_vel_u_bc(cur_state.ghostcells_bottom_vel_u(),
                        cur_state.bndrycells_bottom_vel_u(),
                        [=](const int i) { return cur_state.median_x(i); },
                        [=](const int i) { return cur_state.bottom_y(0); },
                        bottom_vel_u_bc_.second, cur_state.dy(),
                        bottom_vel_u_bc_.first),
        top_vel_u_bc(cur_state.ghostcells_top_vel_u(),
                     cur_state.bndrycells_top_vel_u(),
                     [=](const int i) { return cur_state.median_x(i); },
                     [=](const int i) {
                       return cur_state.top_y(cur_state.cells_y() - 1);
                     },
                     top_vel_u_bc_.second, cur_state.dy(), top_vel_u_bc_.first),
        left_vel_u_bc(cur_state.ghostcells_left_vel_u(),
                      cur_state.bndrycells_right_vel_u(),
                      [=](const int j) { return cur_state.left_x(0); },
                      [=](const int j) { return cur_state.median_y(j); },
                      left_vel_u_bc_.second, cur_state.dx(),
                      left_vel_u_bc_.first),
        right_vel_u_bc(cur_state.ghostcells_right_vel_u(),
                       cur_state.bndrycells_right_vel_u(),
                       [=](const int j) {
                         return cur_state.right_x(cur_state.cells_x() - 1);
                       },
                       [=](const int j) { return cur_state.median_y(j); },
                       right_vel_u_bc_.second, cur_state.dx(),
                       right_vel_u_bc_.first),

        bottom_vel_v_bc(cur_state.ghostcells_bottom_vel_v(),
                        cur_state.bndrycells_bottom_vel_v(),
                        [=](const int i) { return cur_state.median_x(i); },
                        [=](const int i) { return cur_state.bottom_y(0); },
                        bottom_vel_v_bc_.second, cur_state.dy(),
                        bottom_vel_v_bc_.first),
        top_vel_v_bc(cur_state.ghostcells_top_vel_v(),
                     cur_state.bndrycells_top_vel_v(),
                     [=](const int i) { return cur_state.median_x(i); },
                     [=](const int i) {
                       return cur_state.top_y(cur_state.cells_y() - 1);
                     },
                     top_vel_v_bc_.second, cur_state.dy(), top_vel_v_bc_.first),
        left_vel_v_bc(cur_state.ghostcells_left_vel_v(),
                      cur_state.bndrycells_right_vel_v(),
                      [=](const int j) { return cur_state.left_x(0); },
                      [=](const int j) { return cur_state.median_y(j); },
                      left_vel_v_bc_.second, cur_state.dx(),
                      left_vel_v_bc_.first),
        right_vel_v_bc(cur_state.ghostcells_right_vel_v(),
                       cur_state.bndrycells_right_vel_v(),
                       [=](const int j) {
                         return cur_state.right_x(cur_state.cells_x() - 1);
                       },
                       [=](const int j) { return cur_state.median_y(j); },
                       right_vel_v_bc_.second, cur_state.dx(),
                       right_vel_v_bc_.first) {}
  constexpr Mesh &mesh() noexcept { return cur_state; }
  constexpr real cur_time() const noexcept { return time; }

  constexpr real diffusion() const noexcept { return space.diffusion(); }
  constexpr real reynolds() const noexcept { return space.reynolds(); }
  constexpr real prandtl() const noexcept { return space.prandtl(); }
  constexpr real eckert() const noexcept { return space.eckert(); }

protected:
  void apply_bcs() noexcept {
    left_temp_bc.apply(time);
    right_temp_bc.apply(time);
    top_temp_bc.apply(time);
    bottom_temp_bc.apply(time);

    left_vel_u_bc.apply(time);
    right_vel_u_bc.apply(time);
    // Our source term depends on the top and bottom vel_u
    top_vel_u_bc.apply(time);
    bottom_vel_u_bc.apply(time);

    // Our source term depends on the left and right vel_v
    left_vel_v_bc.apply(time);
    right_vel_v_bc.apply(time);
    top_vel_v_bc.apply(time);
    bottom_vel_v_bc.apply(time);
  }

  Mesh cur_state;
  SpaceDisc space;
  real time;

  BoundaryCondition<Mesh::horiz_view> bottom_temp_bc, top_temp_bc;
  BoundaryCondition<Mesh::vert_view> left_temp_bc, right_temp_bc;

  BoundaryCondition<Mesh::horiz_view> bottom_vel_u_bc, top_vel_u_bc;
  BoundaryCondition<Mesh::vert_view> left_vel_u_bc, right_vel_u_bc;

  BoundaryCondition<Mesh::horiz_view> bottom_vel_v_bc, top_vel_v_bc;
  BoundaryCondition<Mesh::vert_view> left_vel_v_bc, right_vel_v_bc;
};

constexpr unsigned long num_eqns(int num_x_cells, int num_y_cells,
                                 int num_ghost_cells) {
  // The interior of the mesh is a rectangle with area num_x_cells * num_y_cells
  // The ghost cells add num_y_cells on the left and right edges, and
  // num_x_cells on the top and bottom edges, which is just a rectangle without
  // its corners
  return (num_x_cells + 2 * num_ghost_cells) *
             (num_y_cells + 2 * num_ghost_cells) -
         4;
}

template <typename SpaceDisc_, typename Solver_>
class ImplicitEuler : public TimeDisc<SpaceDisc_> {
public:
  using Solver = Solver_;
  using SpaceDisc = SpaceDisc_;

  ImplicitEuler(
      const std::pair<real, real> corner_1,
      const std::pair<real, real> corner_2, const size_t cells_x,
      const size_t cells_y, std::function<triple(real, real)> initial_t_u_v,
      const SpaceDisc &space,
      std::pair<BCType, std::function<real(real, real, real)>> bottom_temp_bc_ =
          {BCType::Dirichlet, [](real, real, real) { return 0.0; }},
      std::pair<BCType, std::function<real(real, real, real)>> top_temp_bc_ =
          {BCType::Dirichlet, [](real, real, real) { return 0.0; }},
      std::pair<BCType, std::function<real(real, real, real)>> left_temp_bc_ =
          {BCType::Dirichlet, [](real, real, real) { return 0.0; }},
      std::pair<BCType, std::function<real(real, real, real)>> right_temp_bc_ =
          {BCType::Dirichlet, [](real, real, real) { return 0.0; }},

      std::pair<BCType, std::function<real(real, real, real)>>
          bottom_vel_u_bc_ = {BCType::Dirichlet,
                              [](real, real, real) { return 0.0; }},
      std::pair<BCType, std::function<real(real, real, real)>> top_vel_u_bc_ =
          {BCType::Dirichlet, [](real, real, real) { return 0.0; }},
      std::pair<BCType, std::function<real(real, real, real)>> left_vel_u_bc_ =
          {BCType::Dirichlet, [](real, real, real) { return 0.0; }},
      std::pair<BCType, std::function<real(real, real, real)>> right_vel_u_bc_ =
          {BCType::Dirichlet, [](real, real, real) { return 0.0; }},

      std::pair<BCType, std::function<real(real, real, real)>>
          bottom_vel_v_bc_ = {BCType::Dirichlet,
                              [](real, real, real) { return 0.0; }},
      std::pair<BCType, std::function<real(real, real, real)>> top_vel_v_bc_ =
          {BCType::Dirichlet, [](real, real, real) { return 0.0; }},
      std::pair<BCType, std::function<real(real, real, real)>> left_vel_v_bc_ =
          {BCType::Dirichlet, [](real, real, real) { return 0.0; }},
      std::pair<BCType, std::function<real(real, real, real)>> right_vel_v_bc_ =
          {BCType::Dirichlet, [](real, real, real) { return 0.0; }})
      : TimeDisc<SpaceDisc>(corner_1, corner_2, cells_x, cells_y, initial_t_u_v,
                            space, bottom_temp_bc_, top_temp_bc_, left_temp_bc_,
                            right_temp_bc_, bottom_vel_u_bc_, top_vel_u_bc_,
                            left_vel_u_bc_, right_vel_u_bc_, bottom_vel_v_bc_,
                            top_vel_v_bc_, left_vel_v_bc_, right_vel_v_bc_),
        solver(num_eqns(cells_x, cells_y, Mesh::ghost_cells)),
        system({num_eqns(cells_x, cells_y, Mesh::ghost_cells),
                num_eqns(cells_x, cells_y, Mesh::ghost_cells)}),
        sol_vec(
            vector::shape_type{num_eqns(cells_x, cells_y, Mesh::ghost_cells)}) {
    // Most of our matrix will be 0, so use that as the default value
    for (unsigned long i = 0; i < system.shape()[0]; i++) {
      sol_vec(i) = 0.0;
      for (unsigned long j = 0; j < system.shape()[1]; j++) {
        system(i, j) = 0.0;
      }
    }
    // The mesh velocity components are constant, so the matrix entries are also
    // constant, so just initialize it here after applying the velocity bcs
    this->apply_bcs();
    for (int i = 0; i < this->cur_state.cells_x(); i++) {
      for (int j = 0; j < this->cur_state.cells_y(); j++) {
        // Our matrix is as follows:
        // bottom BC, 0.0, 0.0, ..., 1.0, 0, 0,   ...,                    0, 0,
        // left BC, ... 0, left BC, 0, 0,                    ..., 0, 0, left BC,
        // ...
        // .,                                   .
        // .,                                    .
        // .,                                     .
        // Dx-1, 0, 0, ..., 0, 0, Dy-1, I + Dx0 + Dy0, Dy+1, 0, 0, ..., 0, 0,
        // Dx+1,    ... 0, Dx-1, 0, ..., 0, 0, Dy-1, I + Dx0 + Dy0, Dy+1, 0, 0,
        // ..., 0, 0, Dx+1,    ...
        //
        // Ignore the ghost cell boundary conditions for now
        const int cur_idx = cell_idx(i, j);
        const int left_idx = cell_idx(i - 1, j);
        const int right_idx = cell_idx(i + 1, j);
        const int above_idx = cell_idx(i, j + 1);
        const int below_idx = cell_idx(i, j - 1);
        system(cur_idx, below_idx) = this->space.Dy_m1(this->cur_state, i, j);
        system(cur_idx, above_idx) = this->space.Dy_p1(this->cur_state, i, j);

        system(cur_idx, left_idx) = this->space.Dx_m1(this->cur_state, i, j);
        system(cur_idx, right_idx) = this->space.Dx_p1(this->cur_state, i, j);
      }
    }
  }

  real timestep(const real dt) noexcept {
    assemble_system(dt);
    real max_delta = 0.0;
    const vector &delta = solver.solve(system, sol_vec);
    for (int i = 0; i < this->cur_state.cells_x(); i++) {
      for (int j = 0; j < this->cur_state.cells_y(); j++) {
        this->cur_state.Temp(i, j) += delta(cell_idx(i, j));
        max_delta = std::max(std::abs(delta(cell_idx(i, j))), max_delta);
      }
    }
    this->time += dt;
    return max_delta;
  }

  std::pair<const matrix &, const vector &> assemble_system(const real dt) {
    this->apply_bcs();

    for (int i = 0; i < this->cur_state.cells_x(); i++) {
      for (int j = 0; j < this->cur_state.cells_y(); j++) {
        const int cur_idx = cell_idx(i, j);
        system(cur_idx, cur_idx) = this->space.Dx_0(this->cur_state, i, j) +
                                   this->space.Dy_0(this->cur_state, i, j) +
                                   1.0 / dt;

        sol_vec(cur_idx) = this->space.flux_integral(this->cur_state, i, j) +
                           this->space.source_fd(this->cur_state, i, j);
      }
    }
    // Fill in the ghost cell equations
    // First the top and bottom ghost cells
    for (int i = 0; i < this->cur_state.cells_x(); i++) {
      const int top_ghost_j = this->cur_state.cells_y();
      const int top_ghost_idx = cell_idx(i, top_ghost_j);
      sol_vec(top_ghost_idx) = 0.0;
      system(top_ghost_idx, cell_idx(i, top_ghost_j - 1)) = 1.0;
      system(top_ghost_idx, top_ghost_idx) =
          this->top_temp_bc.implicit_euler_term();

      const int bottom_ghost_j = -this->cur_state.ghost_cells;
      const int bottom_ghost_idx = cell_idx(i, bottom_ghost_j);
      sol_vec(bottom_ghost_idx) = 0.0;
      system(bottom_ghost_idx, cell_idx(i, bottom_ghost_j + 1)) = 1.0;
      system(bottom_ghost_idx, bottom_ghost_idx) =
          this->bottom_temp_bc.implicit_euler_term();
    }
    // Next the left and right ones
    for (int j = 0; j < this->cur_state.cells_y(); j++) {
      const int left_ghost_i = -this->cur_state.ghost_cells;
      const int left_ghost_idx = cell_idx(left_ghost_i, j);
      sol_vec(left_ghost_idx) = 0.0;
      system(left_ghost_idx, cell_idx(left_ghost_i + 1, j)) = 1.0;
      system(left_ghost_idx, left_ghost_idx) =
          this->left_temp_bc.implicit_euler_term();

      const int right_ghost_i = this->cur_state.cells_x();
      const int right_ghost_idx = cell_idx(right_ghost_i, j);
      sol_vec(right_ghost_idx) = 0.0;
      system(right_ghost_idx, cell_idx(right_ghost_i - 1, j)) = 1.0;
      system(right_ghost_idx, right_ghost_idx) =
          this->left_temp_bc.implicit_euler_term();
    }
    return {system, sol_vec};
  }

  unsigned long cell_idx(const int i, const int j) {
    // The left ghost cells number from 0 to mesh.cells_y()
    // Then the next column numbers from mesh.cells_y() to
    // 2 * mesh.cells_y() + 2 * ghost_cells
    // and then from 2 * mesh.cells_y() + 2 * ghost_cells to
    // 3 * this->cur_state.cells_y() + 4 * ghost_cells
    if (i == -this->cur_state.ghost_cells) {
      // The corners are not included
      assert(j >= 0);
      assert(j < this->cur_state.cells_y());
      return j;
    } else if (i == this->cur_state.cells_x()) {
      assert(j >= 0);
      assert(j < this->cur_state.cells_y());
      const int idx =
          this->cur_state.cells_y() +
          i * (this->cur_state.cells_y() + 2 * this->cur_state.ghost_cells) + j;
      assert(idx >= 0);
      assert(idx < num_eqns(this->cur_state.cells_x(),
                            this->cur_state.cells_y(),
                            this->cur_state.ghost_cells));
      return idx;
    } else {
      assert(j >= -this->cur_state.ghost_cells);
      assert(j < this->cur_state.cells_y() + this->cur_state.ghost_cells);
      const int idx =
          this->cur_state.cells_y() +
          i * (this->cur_state.cells_y() + 2 * this->cur_state.ghost_cells) +
          this->cur_state.ghost_cells + j;
      assert(idx >= 0);
      assert(idx < num_eqns(this->cur_state.cells_x(),
                            this->cur_state.cells_y(),
                            this->cur_state.ghost_cells));
      return idx;
    }
  }

  const matrix &sys() const noexcept { return system; }
  const vector &sol() const noexcept { return sol_vec; }
  const vector &delta() const noexcept { return solver.prev_result(); }

protected:
  Solver solver;
  matrix system;
  vector sol_vec;
};

template <typename SpaceDisc_>
class ImplicitEulerApproxFact : public TimeDisc<SpaceDisc_> {
public:
  using Solver = SparseTriDiagSolver;
  using SpaceDisc = SpaceDisc_;

  ImplicitEulerApproxFact(
      const std::pair<real, real> corner_1,
      const std::pair<real, real> corner_2, const size_t cells_x,
      const size_t cells_y, std::function<triple(real, real)> initial_t_u_v,
      const SpaceDisc &space,
      std::pair<BCType, std::function<real(real, real, real)>> bottom_temp_bc_ =
          {BCType::Dirichlet, [](real, real, real) { return 0.0; }},
      std::pair<BCType, std::function<real(real, real, real)>> top_temp_bc_ =
          {BCType::Dirichlet, [](real, real, real) { return 0.0; }},
      std::pair<BCType, std::function<real(real, real, real)>> left_temp_bc_ =
          {BCType::Dirichlet, [](real, real, real) { return 0.0; }},
      std::pair<BCType, std::function<real(real, real, real)>> right_temp_bc_ =
          {BCType::Dirichlet, [](real, real, real) { return 0.0; }},

      std::pair<BCType, std::function<real(real, real, real)>>
          bottom_vel_u_bc_ = {BCType::Dirichlet,
                              [](real, real, real) { return 0.0; }},
      std::pair<BCType, std::function<real(real, real, real)>> top_vel_u_bc_ =
          {BCType::Dirichlet, [](real, real, real) { return 0.0; }},
      std::pair<BCType, std::function<real(real, real, real)>> left_vel_u_bc_ =
          {BCType::Dirichlet, [](real, real, real) { return 0.0; }},
      std::pair<BCType, std::function<real(real, real, real)>> right_vel_u_bc_ =
          {BCType::Dirichlet, [](real, real, real) { return 0.0; }},

      std::pair<BCType, std::function<real(real, real, real)>>
          bottom_vel_v_bc_ = {BCType::Dirichlet,
                              [](real, real, real) { return 0.0; }},
      std::pair<BCType, std::function<real(real, real, real)>> top_vel_v_bc_ =
          {BCType::Dirichlet, [](real, real, real) { return 0.0; }},
      std::pair<BCType, std::function<real(real, real, real)>> left_vel_v_bc_ =
          {BCType::Dirichlet, [](real, real, real) { return 0.0; }},
      std::pair<BCType, std::function<real(real, real, real)>> right_vel_v_bc_ =
          {BCType::Dirichlet, [](real, real, real) { return 0.0; }})
      : TimeDisc<SpaceDisc>(corner_1, corner_2, cells_x, cells_y, initial_t_u_v,
                            space, bottom_temp_bc_, top_temp_bc_, left_temp_bc_,
                            right_temp_bc_, bottom_vel_u_bc_, top_vel_u_bc_,
                            left_vel_u_bc_, right_vel_u_bc_, bottom_vel_v_bc_,
                            top_vel_v_bc_, left_vel_v_bc_, right_vel_v_bc_),
        solver_dx(cells_x + 2 * Mesh::ghost_cells),
        system_dx(matrix::shape_type{cells_x + 2 * Mesh::ghost_cells, 3}),
        sol_vec_dx(vector::shape_type{cells_x + 2 * Mesh::ghost_cells}),
        solver_dy(cells_y + 2 * Mesh::ghost_cells),
        system_dy(matrix::shape_type{cells_y + 2 * Mesh::ghost_cells, 3}),
        sol_vec_dy(vector::shape_type{cells_y + 2 * Mesh::ghost_cells}),
        partial_sol(matrix::shape_type{cells_x, cells_y}) {}

  real timestep(const real dt) noexcept {
    this->apply_bcs();
    // Solve (I + dt Dy) (I + dt Dx) dT = FI + S
    // Start in the Y direction, with constant i
    for (int i = 0; i < this->cur_state.cells_x(); i++) {
      assemble_system_dy(dt, i);
      // Set the ghost cells
      sol_vec_dy(0) = 0.0;
      sol_vec_dy(sol_vec_dy.shape()[0] - 1) = 0.0;
      // The interior cells are given by the flux integral and source term
      for (int j = 0; j < this->cur_state.cells_y(); j++) {
        sol_vec_dy(j + Mesh::ghost_cells) =
            this->space.flux_integral(this->cur_state, i, j) +
            this->space.source_fd(this->cur_state, i, j);
      }
      const vector &col_sol = solver_dy.solve(system_dy, sol_vec_dy);
      // We don't actually care about the dT values for the boundary conditions,
      // so ignore them
      for (int j = 0; j < this->cur_state.cells_y(); j++) {
        partial_sol(i, j) = col_sol(j + Mesh::ghost_cells);
      }
    }
    // Now move on to the X direction, with constant j
    real max_delta = 0.0;
    for (int j = 0; j < this->cur_state.cells_y(); j++) {
      assemble_system_dx(dt, j);
      // Set the ghost cells
      sol_vec_dx(0) = 0.0;
      sol_vec_dx(sol_vec_dx.shape()[0] - 1) = 0.0;
      // The interior cells are given by the previous solution
      for (int i = 0; i < this->cur_state.cells_x(); i++) {
        sol_vec_dx(i + Mesh::ghost_cells) = partial_sol(i, j);
      }
      const vector &row_sol = solver_dx.solve(system_dx, sol_vec_dx);
      for (int i = 0; i < this->cur_state.cells_x(); i++) {
        max_delta = std::max(std::abs(row_sol(i)), max_delta);
        this->cur_state.Temp(i, j) += row_sol(i + Mesh::ghost_cells);
      }
    }
    this->time += dt;
    return max_delta;
  }

  const matrix &assemble_system_dy(const real dt, const int i) {
    // Start with the ghost cells
    system_dy(0, 0) = s_nan; // Should never be used
    system_dy(0, 1) = 1.0;
    system_dy(0, 2) = this->bottom_temp_bc.implicit_euler_term();

    system_dy(system_dy.shape()[0] - 1, 0) =
        this->top_temp_bc.implicit_euler_term();
    system_dy(system_dy.shape()[0] - 1, 1) = 1.0;
    system_dy(system_dy.shape()[0] - 1, 2) = s_nan;

    // Now the interior cells
    for (unsigned long j = 1; j < system_dy.shape()[0] - 1; j++) {
      system_dy(j, 0) = this->space.Dy_m1(this->cur_state, i, j) * dt;
      system_dy(j, 1) = this->space.Dy_0(this->cur_state, i, j) * dt + 1.0;
      system_dy(j, 2) = this->space.Dy_p1(this->cur_state, i, j) * dt;
    }
    return system_dy;
  }

  const matrix &assemble_system_dx(const real dt, const int j) {
    // Start with the ghost cells
    system_dx(0, 0) = s_nan; // Should never be used
    system_dx(0, 1) = 1.0;
    system_dx(0, 2) = this->left_temp_bc.implicit_euler_term();

    system_dx(system_dx.shape()[0] - 1, 0) =
        this->right_temp_bc.implicit_euler_term();
    system_dx(system_dx.shape()[0] - 1, 1) = 1.0;
    system_dx(system_dx.shape()[0] - 1, 2) = s_nan;

    // Now the interior cells
    for (unsigned long i = 1; i < system_dx.shape()[0] - 1; i++) {
      system_dx(i, 0) = this->space.Dx_m1(this->cur_state, i, j) * dt;
      system_dx(i, 1) = this->space.Dx_0(this->cur_state, i, j) * dt + 1.0;
      system_dx(i, 2) = this->space.Dx_p1(this->cur_state, i, j) * dt;
    }
    return system_dx;
  }

protected:
  Solver solver_dx;
  matrix system_dx;
  vector sol_vec_dx;
  Solver solver_dy;
  matrix system_dy;
  vector sol_vec_dy;

  matrix partial_sol;
};

#endif // _TIME_DISC_HPP_

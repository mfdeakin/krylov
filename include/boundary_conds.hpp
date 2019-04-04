
#ifndef _BOUNDARY_CONDS_HPP_
#define _BOUNDARY_CONDS_HPP_

#include <algorithm>
#include <functional>

#include "xtensor/xstrided_view.hpp"
#include "xtensor/xtensor.hpp"
#include "xtensor/xview.hpp"

#include "constants.hpp"

static real homogeneous(real, real, real) { return 0.0; }
static real null_coord(int) { return 0.0; }

enum class BCType { Dirichlet, Neumann };

// This implements a Robin type boundary condition, with a weight factor
// enabling it to implement as a Neumann or Dirichlet boundary condition
//
// Due to the xview having too many possible types, template the boundary
// conditions implementation on it
template <typename viewt> class BoundaryCondition {
public:
  // A boundary condition function takes the x, y, and time values,
  // and returns the value at the interface
  using bc_function = std::function<real(real, real, real)>;
  using cell_coord_function = std::function<real(int)>;

  BoundaryCondition(const viewt &ghost_cells, const viewt &bndry_cells,
                    const cell_coord_function &bndry_x,
                    const cell_coord_function &bndry_y, const bc_function &bc,
                    const real size, const real weight)
      : ghost_cells_(ghost_cells), bndry_cells_(bndry_cells), bc_(bc),
        bndry_x_(bndry_x), bndry_y_(bndry_y), size_(size), weight_(weight) {}

  BoundaryCondition(const viewt &ghost_cells, const viewt &bndry_cells,
                    const cell_coord_function &bndry_x,
                    const cell_coord_function &bndry_y, const bc_function &bc,
                    const real size, const BCType type)
      : ghost_cells_(ghost_cells), bndry_cells_(bndry_cells), bc_(bc),
        bndry_x_(bndry_x), bndry_y_(bndry_y), size_(size),
        weight_(type == BCType::Dirichlet ? 1.0 : 0.0) {}

  BoundaryCondition(const viewt &ghost_cells, const viewt &bndry_cells,
                    const real size, const real weight)
      : BoundaryCondition(ghost_cells, bndry_cells, null_coord, null_coord,
                          homogeneous, size, weight) {}

  BoundaryCondition(const viewt &ghost_cells, const viewt &bndry_cells,
                    const real size, const BCType type)
      : BoundaryCondition(ghost_cells, bndry_cells, null_coord, null_coord,
                          homogeneous, size, type) {}

  real ghost_cell(const real bndry_val, const real x, const real y,
                  const real time) const noexcept {
    return (2.0 * bc_(x, y, time) - bndry_val) * weight_ +
           (size_ * bc_(x, y, time) + bndry_val) * (1.0 - weight_);
  }

  void apply(const real time) noexcept {
    for (int idx = 0; idx < bndry_cells_.shape()[0]; idx++) {
      const real x = bndry_x_(idx);
      const real y = bndry_y_(idx);
      const real gc_val = ghost_cell(bndry_cells_(idx), x, y, time);
      ghost_cells_(idx) = gc_val;
    }
  }

protected:
  viewt ghost_cells_;
  viewt bndry_cells_;
  bc_function bc_;
  cell_coord_function bndry_x_;
  cell_coord_function bndry_y_;
  real size_, weight_;
};

template <typename viewt> class DirichletBC : public BoundaryCondition<viewt> {
public:
  using BaseType = BoundaryCondition<viewt>;
  using bc_function = typename BaseType::bc_function;
  using cell_coord_function = typename BaseType::cell_coord_function;

  DirichletBC(const viewt &ghost_cells, const viewt &bndry_cells,
              const cell_coord_function &bndry_x,
              const cell_coord_function &bndry_y, const bc_function &bc)
      : BaseType(bndry_cells, ghost_cells, bndry_x, bndry_y, bc, 0.0, 1.0) {}

  DirichletBC(const viewt &ghost_cells, const viewt &bndry_cells)
      : BaseType(ghost_cells, bndry_cells, 0.0, 1.0) {}
};

template <typename viewt> class NeumannBC : public BoundaryCondition<viewt> {
public:
  using BaseType = BoundaryCondition<viewt>;
  using bc_function = typename BaseType::bc_function;
  using cell_coord_function = typename BaseType::cell_coord_function;

  NeumannBC(const viewt &ghost_cells, const viewt &bndry_cells,
            const cell_coord_function &bndry_x,
            const cell_coord_function &bndry_y, const bc_function &bc,
            const real size)
      : BaseType(ghost_cells, bndry_cells, bndry_x, bndry_y, bc, size, 0.0) {}

  // For homogeneous bcs, the size parameter doesn't have any impact
  NeumannBC(const viewt &ghost_cells, const viewt &bndry_cells,
            const real size = 0.0)
      : BaseType(ghost_cells, bndry_cells, 0.0, 0.0) {}
};

#endif // _BOUNDARY_CONDS_HPP_

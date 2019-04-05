
#ifndef _MESH_HPP_
#define _MESH_HPP_

#include <functional>
#include <iterator>

#include "constants.hpp"

#include "xtensor/xtensor.hpp"
#include "xtensor/xview.hpp"

class Mesh {
public:
  // Assume a single ghost cell on each side of the boundary
  static constexpr int ghost_cells = 1;

  // Cells are (externally) zero referenced from the bottom left corners
  // interior cell
  constexpr real left_x(int cell_x) const noexcept {
    return min_x_ + cell_x * dx_;
  }

  constexpr real right_x(int cell_x) const noexcept {
    return min_x_ + (cell_x + 1) * dx_;
  }

  constexpr real bottom_y(int cell_y) const noexcept {
    return min_y_ + cell_y * dy_;
  }

  constexpr real top_y(int cell_y) const noexcept {
    return min_y_ + (cell_y + 1) * dy_;
  }

  constexpr real median_x(int cell_x) const noexcept {
    return min_x_ + dx_ / 2 + cell_x * dx_;
  }

  constexpr real median_y(int cell_y) const noexcept {
    return min_y_ + dy_ / 2 + cell_y * dy_;
  }

  constexpr int x_idx(real x) const noexcept {
    return static_cast<int>(x / dx_ - min_x_ / dx_);
  }

  constexpr int y_idx(real y) const noexcept {
    return static_cast<int>(y / dy_ - min_y_ / dy_);
  }

  constexpr real dx() const noexcept { return dx_; }

  constexpr real dy() const noexcept { return dy_; }

  Mesh(const std::pair<real, real> &corner_1,
       const std::pair<real, real> &corner_2, const size_t x_cells,
       const size_t y_cells) noexcept;

  Mesh(const std::pair<real, real> &corner_1,
       const std::pair<real, real> &corner_2, const size_t x_cells,
       const size_t y_cells,
       std::function<triple(real, real)> initial_t_u_v) noexcept;

  Mesh(const std::pair<real, real> &corner_1,
       const std::pair<real, real> &corner_2, const size_t x_cells,
       const size_t y_cells,
       const std::function<real(real, real)> &initial_temp,
       const std::function<real(real, real)> &initial_vel_u,
       const std::function<real(real, real)> &initial_vel_v) noexcept
      : Mesh(corner_1, corner_2, x_cells, y_cells,
             [initial_temp, initial_vel_u, initial_vel_v](real x, real y) {
               return triple{initial_temp(x, y), initial_vel_u(x, y),
                             initial_vel_v(x, y)};
             }) {}

  real Temp(int i, int j) const noexcept {
    assert(i >= -1);
    assert(i <= cells_x());
    assert(j >= -1);
    assert(j <= cells_y());
    return temp_cva_(i + ghost_cells, j + ghost_cells);
  }
  real &Temp(int i, int j) noexcept {
    assert(i >= -1);
    assert(i <= cells_x());
    assert(j >= -1);
    assert(j <= cells_y());
    return temp_cva_(i + ghost_cells, j + ghost_cells);
  }

  real vel_u(int i, int j) const noexcept {
    assert(i >= -1);
    assert(i <= cells_x());
    assert(j >= -1);
    assert(j <= cells_y());
    return vel_u_cva_(i + ghost_cells, j + ghost_cells);
  }
  real &vel_u(int i, int j) noexcept {
    assert(i >= -1);
    assert(i <= cells_x());
    assert(j >= -1);
    assert(j <= cells_y());
    return vel_u_cva_(i + ghost_cells, j + ghost_cells);
  }

  real vel_v(int i, int j) const noexcept {
    assert(i >= -1);
    assert(i <= cells_x());
    assert(j >= -1);
    assert(j <= cells_y());
    return vel_v_cva_(i + ghost_cells, j + ghost_cells);
  }
  real &vel_v(int i, int j) noexcept {
    assert(i >= -1);
    assert(i <= cells_x());
    assert(j >= -1);
    assert(j <= cells_y());
    return vel_v_cva_(i + ghost_cells, j + ghost_cells);
  }

  // The joys of template metaprogramming - code that's selectively (for unknown
  // reasons) extremely sensitive to having the right types
  using tensor_type = xt::xtensor<double, 2>;
  using range_type =
      decltype(xt::range(std::declval<int>(), std::declval<int>()));
  using vert_view = decltype(xt::view(std::declval<tensor_type &>(),
                                      std::declval<unsigned long>(),
                                      std::declval<range_type>()));
  using horiz_view = decltype(xt::view(std::declval<tensor_type &>(),
                                       std::declval<range_type>(),
                                       std::declval<unsigned long>()));

  vert_view ghostcells_left_Temp() noexcept {
    // An example of the selective sensitivity of templates to the types
    return xt::view(temp_cva_, static_cast<unsigned long>(0),
                    xt::range(1, temp_cva_.shape()[1] - 1));
  }
  vert_view ghostcells_right_Temp() noexcept {
    return xt::view(temp_cva_, temp_cva_.shape()[0] - 1,
                    xt::range(1, temp_cva_.shape()[1] - 2));
  }
  horiz_view ghostcells_top_Temp() noexcept {
    return xt::view(temp_cva_, xt::range(1, temp_cva_.shape()[0] - 1),
                    temp_cva_.shape()[1] - 1);
  }
  horiz_view ghostcells_bottom_Temp() noexcept {
    return xt::view(temp_cva_, xt::range(1, temp_cva_.shape()[0] - 1),
                    static_cast<unsigned long>(0));
  }

  vert_view ghostcells_left_vel_u() noexcept {
    return xt::view(vel_u_cva_, static_cast<unsigned long>(0),
                    xt::range(1, vel_u_cva_.shape()[1] - 1));
  }
  vert_view ghostcells_right_vel_u() noexcept {
    return xt::view(vel_u_cva_, vel_u_cva_.shape()[0] - 1,
                    xt::range(1, vel_u_cva_.shape()[1] - 1));
  }
  horiz_view ghostcells_top_vel_u() noexcept {
    return xt::view(vel_u_cva_, xt::range(1, vel_u_cva_.shape()[0] - 1),
                    vel_u_cva_.shape()[1] - 1);
  }
  horiz_view ghostcells_bottom_vel_u() noexcept {
    return xt::view(vel_u_cva_, xt::range(1, vel_u_cva_.shape()[0] - 1),
                    static_cast<unsigned long>(0));
  }

  vert_view ghostcells_left_vel_v() noexcept {
    return xt::view(vel_v_cva_, static_cast<unsigned long>(0),
                    xt::range(1, vel_v_cva_.shape()[1] - 1));
  }
  vert_view ghostcells_right_vel_v() noexcept {
    return xt::view(vel_v_cva_, vel_v_cva_.shape()[0] - 1,
                    xt::range(1, vel_v_cva_.shape()[1] - 1));
  }
  horiz_view ghostcells_top_vel_v() noexcept {
    return xt::view(vel_v_cva_, xt::range(1, vel_v_cva_.shape()[0] - 1),
                    vel_v_cva_.shape()[1] - 1);
  }
  horiz_view ghostcells_bottom_vel_v() noexcept {
    return xt::view(vel_v_cva_, xt::range(1, vel_v_cva_.shape()[0] - 1),
                    static_cast<unsigned long>(0));
  }

  vert_view bndrycells_left_Temp() noexcept {
    return xt::view(temp_cva_, static_cast<unsigned long>(1),
                    xt::range(1, temp_cva_.shape()[1] - 1));
  }
  vert_view bndrycells_right_Temp() noexcept {
    return xt::view(temp_cva_, temp_cva_.shape()[0] - 2,
                    xt::range(1, temp_cva_.shape()[1] - 1));
  }
  horiz_view bndrycells_top_Temp() noexcept {
    return xt::view(temp_cva_, xt::range(1, temp_cva_.shape()[0] - 1),
                    temp_cva_.shape()[1] - 2);
  }
  horiz_view bndrycells_bottom_Temp() noexcept {
    return xt::view(temp_cva_, xt::range(1, temp_cva_.shape()[0] - 1),
                    static_cast<unsigned long>(1));
  }

  vert_view bndrycells_left_vel_u() noexcept {
    return xt::view(vel_u_cva_, static_cast<unsigned long>(1),
                    xt::range(1, vel_u_cva_.shape()[1] - 1));
  }
  vert_view bndrycells_right_vel_u() noexcept {
    return xt::view(vel_u_cva_, vel_u_cva_.shape()[0] - 2,
                    xt::range(1, vel_u_cva_.shape()[1] - 1));
  }
  horiz_view bndrycells_top_vel_u() noexcept {
    return xt::view(vel_u_cva_, xt::range(1, vel_u_cva_.shape()[0] - 1),
                    vel_u_cva_.shape()[1] - 2);
  }
  horiz_view bndrycells_bottom_vel_u() noexcept {
    return xt::view(vel_u_cva_, xt::range(1, vel_u_cva_.shape()[0] - 1),
                    static_cast<unsigned long>(1));
  }

  vert_view bndrycells_left_vel_v() noexcept {
    return xt::view(vel_v_cva_, static_cast<unsigned long>(1),
                    xt::range(1, vel_v_cva_.shape()[1] - 1));
  }
  vert_view bndrycells_right_vel_v() noexcept {
    return xt::view(vel_v_cva_, vel_v_cva_.shape()[0] - 2,
                    xt::range(1, vel_v_cva_.shape()[1] - 1));
  }
  horiz_view bndrycells_top_vel_v() noexcept {
    return xt::view(vel_v_cva_, xt::range(1, vel_v_cva_.shape()[0] - 1),
                    vel_v_cva_.shape()[1] - 2);
  }
  horiz_view bndrycells_bottom_vel_v() noexcept {
    return xt::view(vel_v_cva_, xt::range(1, vel_v_cva_.shape()[0] - 1),
                    static_cast<unsigned long>(1));
  }

  real interpolate_Temp(real x, real y) const noexcept {
    return interpolate(temp_cva_, x, y);
  }
  real interpolate_vel_u(real x, real y) const noexcept {
    return interpolate(vel_u_cva_, x, y);
  }
  real interpolate_vel_v(real x, real y) const noexcept {
    return interpolate(vel_v_cva_, x, y);
  }

  const real *data_Temp() const noexcept { return temp_cva_.data(); }
  const real *data_vel_u() const noexcept { return vel_u_cva_.data(); }
  const real *data_vel_v() const noexcept { return vel_v_cva_.data(); }

  const xt::xtensor<real, 2> &array_Temp() const noexcept { return temp_cva_; }
  const xt::xtensor<real, 2> &array_vel_u() const noexcept {
    return vel_u_cva_;
  }
  const xt::xtensor<real, 2> &array_vel_v() const noexcept {
    return vel_v_cva_;
  }

  int cells_x() const noexcept {
    return temp_cva_.shape()[0] - 2 * ghost_cells;
  }
  int cells_y() const noexcept {
    return temp_cva_.shape()[1] - 2 * ghost_cells;
  }

protected:
  real interpolate(const xt::xtensor<real, 2> &src, real x, real y) const
      noexcept;

  real min_x_, max_x_, min_y_, max_y_;
  real dx_, dy_;

  xt::xtensor<real, 2> temp_cva_;
  xt::xtensor<real, 2> vel_u_cva_;
  xt::xtensor<real, 2> vel_v_cva_;
};

#endif // _MESH_HPP_

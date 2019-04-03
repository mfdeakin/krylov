
#include <algorithm>
#include <cmath>
#include <limits>

#include "mesh.hpp"

Mesh::Mesh(const std::pair<real, real> &corner_1,
           const std::pair<real, real> &corner_2, const size_t cells_x,
           const size_t cells_y) noexcept
    : min_x_(std::min(corner_1.first, corner_2.first)),
      max_x_(std::max(corner_1.first, corner_2.first)),
      min_y_(std::min(corner_1.second, corner_2.second)),
      max_y_(std::max(corner_1.second, corner_2.second)),
      dx_((max_x_ - min_x_) / cells_x), dy_((max_y_ - min_y_) / cells_y),
      temp_cva_(std::array<size_t, 2>{cells_x + 2 * ghost_cells,
                                      cells_y + 2 * ghost_cells}),
      vel_u_cva_(std::array<size_t, 2>{cells_x + 2 * ghost_cells,
                                       cells_y + 2 * ghost_cells}),
      vel_v_cva_(std::array<size_t, 2>{cells_x + 2 * ghost_cells,
                                       cells_y + 2 * ghost_cells}) {
  assert(!std::isnan(min_x_) && !std::isnan(max_x_));
  assert(!std::isnan(min_y_) && !std::isnan(max_y_));
  assert(min_x_ != max_x_);
  assert(min_y_ != max_y_);
  // xtensor doesn't initialize the arrays, set all of the interior cells to 0
  for (int i = 0; i < this->cells_x(); i++) {
    for (int j = 0; j < this->cells_y(); j++) {
      Temp(i, j) = 0.0;
      vel_u(i, j) = 0.0;
      vel_v(i, j) = 0.0;
    }
  }
  // For safety, set the ghost cells to a signalling NaN, so if they are used in
  // an arithmetic calculation, a signal is sent
  for (int i = -1; i < this->cells_x() + 1; i++) {
    Temp(i, -1) = s_nan;
    Temp(i, this->cells_y()) = s_nan;
    vel_u(i, -1) = s_nan;
    vel_u(i, this->cells_y()) = s_nan;
    vel_v(i, -1) = s_nan;
    vel_v(i, this->cells_y()) = s_nan;
  }
  for (int j = -1; j < this->cells_y() + 1; j++) {
    Temp(-1, j) = s_nan;
    Temp(this->cells_x(), j) = s_nan;
    vel_u(-1, j) = s_nan;
    vel_u(this->cells_x(), j) = s_nan;
    vel_v(-1, j) = s_nan;
    vel_v(this->cells_x(), j) = s_nan;
  }
}

Mesh::Mesh(const std::pair<real, real> &corner_1,
           const std::pair<real, real> &corner_2, const size_t cells_x,
           const size_t cells_y, std::function<triple(real, real)> f) noexcept
    : Mesh(corner_1, corner_2, cells_x, cells_y) {
  // Initialize the interior only - the boundaries should be specified by
  // boundary conditions
  for (int i = 0; i < this->cells_x(); i++) {
    for (int j = 0; j < this->cells_y(); j++) {
      auto [t, u, v] = f(median_x(i), median_y(j));
      Temp(i, j) = t;
      vel_u(i, j) = u;
      vel_v(i, j) = v;
    }
  }
}

// Use bilinear interpolation to estimate the value at the requested point
real Mesh::interpolate(const xt::xtensor<real, 2> &src, real x, real y) const
    noexcept {
  const int cur_x_cell = x_idx(x);
  const int right_cell = median_x(cur_x_cell) > x ? cur_x_cell : cur_x_cell + 1;
  const int left_cell = right_cell - 1;
  const int cur_y_cell = y_idx(y);
  const int top_cell = median_y(cur_y_cell) > y ? cur_y_cell : cur_y_cell + 1;
  const int bot_cell = top_cell - 1;
  assert(left_cell >= -1);
  assert(right_cell <= cells_x());
  assert(bot_cell >= -1);
  assert(top_cell <= cells_y());
  const real left_weight = (median_x(right_cell) - x) / dx_;
  const real bot_weight = (median_y(top_cell) - y) / dy_;
  return left_weight * (bot_weight * src(left_cell, bot_cell) +
                        (1.0 - bot_weight) * src(left_cell, top_cell)) +
         (1.0 - left_weight) * (bot_weight * src(right_cell, bot_cell) +
                                (1.0 - bot_weight) * src(right_cell, top_cell));
}

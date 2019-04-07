
#include <chrono>
#include <cmath>
#include <random>

#include <gtest/gtest.h>

#include "boundary_conds.hpp"
#include "constants.hpp"
#include "matrix.hpp"
#include "mesh.hpp"
#include "solver.hpp"
#include "space_disc.hpp"
#include "time_disc.hpp"

TEST(mesh, cell_overlap) {
  constexpr int cells_x = 32, cells_y = 32;
  Mesh coarse({0.0, 0.0}, {1.0, 1.0}, cells_x, cells_y);
  Mesh fine({0.0, 0.0}, {1.0, 1.0}, cells_x * 2, cells_y * 2);
  for (int i = 0; i < coarse.cells_x(); i++) {
    EXPECT_NEAR(coarse.median_x(i), fine.right_x(i * 2),
                4.0 * std::numeric_limits<real>::epsilon() *
                    coarse.median_x(i));
    EXPECT_NEAR(coarse.median_x(i), fine.left_x(i * 2 + 1),
                4.0 * std::numeric_limits<real>::epsilon() *
                    coarse.median_x(i));
    EXPECT_NEAR(coarse.left_x(i), fine.left_x(i * 2),
                4.0 * std::numeric_limits<real>::epsilon() * coarse.left_x(i));
    EXPECT_NEAR(coarse.right_x(i), fine.right_x(i * 2 + 1),
                4.0 * std::numeric_limits<real>::epsilon() * coarse.right_x(i));
  }
  EXPECT_EQ(2 * coarse.cells_y(), fine.cells_y());
  for (int j = 0; j < coarse.cells_y(); j++) {
    EXPECT_NEAR(coarse.median_y(j), fine.top_y(j * 2),
                4.0 * std::numeric_limits<real>::epsilon() *
                    coarse.median_y(j));
    EXPECT_NEAR(coarse.median_y(j), fine.bottom_y(j * 2 + 1),
                4.0 * std::numeric_limits<real>::epsilon() *
                    coarse.median_y(j));
    EXPECT_NEAR(coarse.bottom_y(j), fine.bottom_y(j * 2),
                4.0 * std::numeric_limits<real>::epsilon() *
                    coarse.bottom_y(j));
    EXPECT_NEAR(coarse.top_y(j), fine.top_y(j * 2 + 1),
                4.0 * std::numeric_limits<real>::epsilon() * coarse.top_y(j));
  }
}

ImplicitEuler<SecondOrderCentered, LUSolver>
default_ie_lu_solver(const int cells_x, const int cells_y) {
  constexpr real reynolds = 25.0;
  constexpr real prandtl = 0.7;
  constexpr real eckert = 0.1;
  std::pair<real, real> c1{0.0, 0.0};
  std::pair<real, real> c2{40.0, 1.0};
  SecondOrderCentered space_disc(1.0, reynolds, prandtl, eckert);

  std::pair<BCType, std::function<real(real, real, real)>> def_bc{
      BCType::Dirichlet,
      [](const real, const real, const real) { return 0.0; }};
  std::pair<BCType, std::function<real(real, real, real)>> left_temp{
      BCType::Dirichlet,
      [](const real, const real y, const real) { return y; }};
  std::pair<BCType, std::function<real(real, real, real)>> bottom_temp{
      BCType::Dirichlet,
      [](const real, const real, const real) { return 0.0; }};
  std::pair<BCType, std::function<real(real, real, real)>> top_temp{
      BCType::Dirichlet,
      [](const real, const real, const real) { return 1.0; }};

  ImplicitEuler<SecondOrderCentered, LUSolver> sim(
      c1, c2, cells_x, cells_y,
      [](const real x, const real y) {
        constexpr real avg_u = 3.0;
        return triple{0.0, 6.0 * avg_u * y * (1.0 - y), 0.0};
      },
      space_disc, bottom_temp, top_temp, left_temp, def_bc, def_bc, def_bc,
      def_bc, def_bc, def_bc, def_bc, def_bc, def_bc);
  return sim;
}

TEST(cell_indices, implicit_euler) {
  constexpr int cells_x = 18, cells_y = 8;
  ImplicitEuler<SecondOrderCentered, LUSolver> sim =
      default_ie_lu_solver(cells_x, cells_y);
  // Better would be to ensure that the cell indices are all within range and
  // unique, but this is easier for now
  int expected_idx = 0;
  for (int j = 0; j < cells_y; j++) {
    EXPECT_EQ(sim.cell_idx(-1, j), expected_idx);
    expected_idx++;
  }
  for (int i = 0; i < cells_x; i++) {
    for (int j = -1; j <= cells_y; j++) {
      EXPECT_EQ(sim.cell_idx(i, j), expected_idx);
      expected_idx++;
    }
  }
  for (int j = 0; j < cells_y; j++) {
    EXPECT_EQ(sim.cell_idx(cells_x, j), expected_idx);
    expected_idx++;
  }
}

std::pair<matrix, vector> simple_test(const unsigned long size) {
  matrix A(matrix::shape_type{size, size});
  vector b(vector::shape_type{size});
  for (unsigned long i = 0; i < A.shape()[0]; i++) {
    b(i) = 0.0;
    for (unsigned long j = 0; j < A.shape()[1]; j++) {
      A(i, j) = 0.0;
    }
  }
  b(4) = 1.0;
  b(5) = 5.0;
  b(6) = 1.0;
  for (unsigned long i = 0; i < A.shape()[0]; i++) {
    A(i, i) = -2.0;
    if (i < A.shape()[0] - 1) {
      A(i + 1, i) = 1.0;
      A(i, i + 1) = 1.0;
    }
  }
  return {A, b};
}

TEST(tridiag, solvers) {
  constexpr unsigned long size = 10;
  auto [A, b] = simple_test(size);
  TriDiagSolver solver(size);
  vector x = solver.solve(A, b);
  vector expected{5.0,         10.0, 15.0, 20.0, 25.0,
                  199.0 / 7.0, 24.0, 18.0, 12.0, 6.0};
  expected *= -7.0 / 11.0;
  for (unsigned long i = 0; i < size; i++) {
    EXPECT_NEAR(x(i), expected(i), 1e-14);
  }
}

TEST(sparsetridiag, solvers) {
  constexpr unsigned long size = 10;
  const auto [A, b] = simple_test(size);
  matrix Asparse(matrix::shape_type{A.shape()[0], 3});
  for (unsigned long i = 0; i < Asparse.shape()[0]; i++) {
    Asparse(i, 1) = A(i, i);
    if (i < Asparse.shape()[0] - 1) {
      Asparse(i, 2) = A(i, i + 1);
    }
    if (i > 0) {
      Asparse(i, 0) = A(i, i - 1);
    }
  }
  Asparse(0, 0) = s_nan;
  Asparse(size - 1, 2) = s_nan;
  SparseTriDiagSolver solver(size);
  vector x = solver.solve(Asparse, b);
  vector expected{5.0,         10.0, 15.0, 20.0, 25.0,
                  199.0 / 7.0, 24.0, 18.0, 12.0, 6.0};
  expected *= -7.0 / 11.0;
  for (unsigned long i = 0; i < size; i++) {
    EXPECT_NEAR(x(i), expected(i), 1e-14);
  }
}

TEST(arnoldi, gmres) {
  constexpr unsigned long size = 10;
  auto [A, b] = simple_test(size);
  // This subspace will fully solve the system, slower than LU would have...
  // It converges slowly enough that doing any less makes testing difficult
  GMRESSolver<10> test(10);
  test.initial_subspace(b);
  {
    const vector &check = test.subspace_vector(0);
    for (unsigned long i = 0; i < size; i++) {
      EXPECT_NEAR(check(i), b(i) * std::sqrt(3) / 9.0, 1e-15);
    }
  }
  {
    const real h10 = test.add_arnoldi(A);
    EXPECT_NEAR(h10, 2.0 * std::sqrt(278) / 27.0, 1e-15);
    const vector &check = test.subspace_vector(1);
    const real scale = std::sqrt(3.0 * 278) / 5004.0;
    const vector expected_1{0.0,   0.0,   0.0,  27.0, 115.0,
                            -46.0, 115.0, 27.0, 0.0,  0.0};
    for (unsigned long i = 0; i < size; i++) {
      EXPECT_NEAR(check(i), scale * expected_1(i), 1e-15);
    }
  }
  {
    const real h21 = test.add_arnoldi(A);
    EXPECT_NEAR(h21, 3.0 * std::sqrt(2373.0) / 139.0, 1e-15);
    const vector &check = test.subspace_vector(2);
    const real scale = std::sqrt(278.0 * 791.0) / 879592.0;
    const vector expected_2{0.0,   0.0,    278.0,  1265.0, -275.0,
                            110.0, -275.0, 1265.0, 278.0,  0.0};
    for (unsigned long i = 0; i < size; i++) {
      EXPECT_NEAR(check(i) / scale, expected_2(i), 2e-12);
    }
  }
  {
    const vector &x = test.solve(A, b);
    vector expected{5.0,         10.0, 15.0, 20.0, 25.0,
                    199.0 / 7.0, 24.0, 18.0, 12.0, 6.0};
    expected *= -7.0 / 11.0;
    for (unsigned long i = 0; i < size; i++) {
      EXPECT_NEAR(x(i), expected(i), 1e-12);
    }
  }
}

TEST(solve_20_10, implicit_euler) {
  constexpr int cells_x = 18, cells_y = 8;
  ImplicitEuler<SecondOrderCentered, LUSolver> sim =
      default_ie_lu_solver(cells_x, cells_y);
  real delta = std::numeric_limits<real>::infinity();
  auto start = std::chrono::high_resolution_clock::now();
  int num_ts = 0;
  while (delta > 1e-9) {
    delta = sim.timestep(0.25);
    num_ts++;
    // std::cout << delta << " delta" << std::endl;
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::cout << "Time to steady state: "
            << std::chrono::duration_cast<std::chrono::microseconds>(end -
                                                                     start)
                   .count()
            << " us; " << num_ts << " iterations; " << delta << " final delta"
            << std::endl;
}

TEST(lu_decomp, solvers) {
  matrix m1{{2.0, 3.0, 4.0, 20.0},
            {5.0, 6.0, 7.0, 30.0},
            {8.0, 9.0, 20.0, 40.0},
            {11.0, 12.0, 130.0, 160.0}};
  LUSolver solver(m1.shape()[0]);
  auto [l, u] = solver.lu_decomp(m1);
  EXPECT_EQ(l.shape()[0], m1.shape()[0]);
  EXPECT_EQ(l.shape()[1], m1.shape()[1]);
  EXPECT_EQ(u.shape()[0], m1.shape()[0]);
  EXPECT_EQ(u.shape()[1], m1.shape()[1]);
  matrix ltu(m1.shape());
  dot(ltu, l, u);
  for (unsigned long i = 0; i < l.shape()[0]; i++) {
    for (unsigned long j = 0; j < l.shape()[1]; j++) {
      EXPECT_EQ(ltu(i, j), m1(i, j));
      if (j > i) {
        EXPECT_EQ(l(i, j), 0.0);
        EXPECT_EQ(u(j, i), 0.0);
      }
    }
  }
  vector x_expected{{-1.0, 1.0, 2.0, -4.0}};
  vector b(x_expected.shape());
  dot(b, m1, x_expected);
  auto x = solver.solve(m1, b);
  for (unsigned long i = 0; i < b.shape()[0]; i++) {
    EXPECT_EQ(x(i), x_expected(i));
  }
}

TEST(product, matrix) {
  matrix m1{{1.0, -4.0, -2.0}, {1.0, 2.0, 4.0}};
  matrix m2{
      {3.0, 6.0, 9.0, 12.0}, {4.0, 7.0, 10.0, 13.0}, {5.0, 8.0, 11.0, 14.0}};
  matrix expected{{-23.0, -38.0, -53.0, -68.0}, {31.0, 52.0, 73.0, 94.0}};
  matrix result(expected.shape());
  dot(result, m1, m2);
  vector x{2.0, 4.0, 5.0};
  vector y_expected{-24.0, 30.0};
  vector y(y_expected.shape());
  dot(y, m1, x);
  for (unsigned long i = 0; i < m1.shape()[0]; i++) {
    EXPECT_EQ(y_expected(i), y(i));
    for (unsigned long j = 0; j < m2.shape()[1]; j++) {
      EXPECT_EQ(expected(i, j), result(i, j));
    }
  }
}

Mesh construct_mesh_with_hbc(
    const int cells_x, const int cells_y, std::pair<real, real> corner_1,
    std::pair<real, real> corner_2,
    std::function<triple(real, real)> initial_conds) noexcept;

TEST(flux_int_components, second_order_space_disc) {
  constexpr real T0 = 1.0;
  constexpr real u0 = 0.0;
  constexpr real v0 = 0.0;
  constexpr int cells_x = 256, cells_y = 256;
  constexpr std::pair<real, real> corner_1{0.0, 0.0}, corner_2{1.0, 1.0};
  // T = T0 cos(pi x) sin(pi y)
  // u = u0 y sin(pi x)
  // v = v0 x cos(pi y)
  Mesh src(construct_mesh_with_hbc(
      cells_x, cells_y, corner_1, corner_2, [&](real x, real y) {
        return triple{T0 * std::cos(pi * x) * std::sin(pi * y),
                      u0 * y * std::sin(pi * x), v0 * x * std::cos(pi * y)};
      }));
  Mesh fine(construct_mesh_with_hbc(
      2 * cells_x, 2 * cells_y, corner_1, corner_2, [&](real x, real y) {
        return triple{T0 * std::cos(pi * x) * std::sin(pi * y),
                      u0 * y * std::sin(pi * x), v0 * x * std::cos(pi * y)};
      }));

  constexpr real reynolds = q_nan;
  constexpr real prandtl = q_nan;
  constexpr real eckert = q_nan;
  constexpr real diffusion = q_nan;
  SecondOrderCentered sd(diffusion, reynolds, prandtl, eckert);
  const auto expected_dx = [](const real x, const real y) {
    return -T0 * pi * std::sin(pi * x) * std::sin(pi * y);
  };
  const auto expected_dy = [](const real x, const real y) {
    return T0 * pi * std::cos(pi * x) * std::cos(pi * y);
  };

  for (int i = 0; i < src.cells_x(); i++) {
    for (int j = 0; j < src.cells_y(); j++) {
      EXPECT_NEAR(sd.dx_flux(src, i, j),
                  expected_dx(src.right_x(i), src.median_y(j)), 1e-4);
      EXPECT_NEAR(
          sd.dx_flux(fine, 2 * i + 1, 2 * j + 1),
          expected_dx(fine.right_x(2 * i + 1), fine.median_y(2 * j + 1)), 1e-5);

      EXPECT_NEAR(sd.dy_flux(src, i, j),
                  expected_dy(src.median_x(i), src.top_y(j)), 1e-4);
      EXPECT_NEAR(sd.dy_flux(fine, 2 * i + 1, 2 * j + 1),
                  expected_dy(fine.median_x(2 * i + 1), fine.top_y(2 * j + 1)),
                  1e-5);
      // if (sd.dy_flux(src, i, j) != expected_dy &&
      //     sd.dy_flux(fine, 2 * i + 1, 2 * j + 1) != expected_dy) {
      //   EXPECT_GT(
      //       -std::log2(std::abs(
      //           (sd.dy_flux(src, i, j) - expected_dy) /
      //           (sd.dy_flux(fine, 2 * i + 1, 2 * j + 1) - expected_dy))),
      //       2.0 - 1e-6);
      // }
    }
  }
}

TEST(flux_int_diffusion, second_order_space_disc) {
  constexpr real T0 = 1.0;
  constexpr real u0 = 0.0;
  constexpr real v0 = 0.0;
  constexpr int cells_x = 512, cells_y = 512;
  constexpr std::pair<real, real> corner_1{0.0, 0.0}, corner_2{1.0, 1.0};
  // T = T0 cos(pi x) sin(pi y)
  // u = u0 y sin(pi x)
  // v = v0 x cos(pi y)
  Mesh src(construct_mesh_with_hbc(
      cells_x, cells_y, corner_1, corner_2, [&](real x, real y) {
        return triple{T0 * std::cos(pi * x) * std::sin(pi * y),
                      u0 * y * std::sin(pi * x), v0 * x * std::cos(pi * y)};
      }));

  constexpr real reynolds = 50.0;
  constexpr real prandtl = 0.7;
  constexpr real eckert = 0.1;
  {
    // Everything should be 0 in this case
    constexpr real diffusion = 0.0;
    SecondOrderCentered sd(diffusion, reynolds, prandtl, eckert);
    for (int i = 0; i < src.cells_x(); i++) {
      for (int j = 0; j < src.cells_y(); j++) {
        const real x = src.median_x(i);
        const real y = src.median_y(j);
        const real flux_integral = sd.flux_integral(src, i, j);
        // u0 T0 pi cos(2 pi x) y sin(pi y)
        // + v0 T0 pi x cos(pi x) cos(2 pi y)
        // + (2 T0 pi^2) / (Re Pr) cos(pi x) sin(pi y)
        EXPECT_EQ(flux_integral,
                  u0 * T0 * pi * std::cos(2.0 * pi * x) * y * std::sin(pi * y) +
                      v0 * T0 * pi * x * std::cos(pi * x) *
                          std::cos(2.0 * pi * y) +
                      diffusion * (2.0 * T0 * pi * pi) / (reynolds * prandtl) *
                          std::cos(pi * x) * std::sin(pi * y));
      }
    }
  }
  {
    constexpr real diffusion = 1.0;
    SecondOrderCentered sd(diffusion, reynolds, prandtl, eckert);
    for (int i = 0; i < src.cells_x(); i++) {
      for (int j = 0; j < src.cells_y(); j++) {
        const real x = src.median_x(i);
        const real y = src.median_y(j);
        const real flux_integral = sd.flux_integral(src, i, j);
        // u0 T0 pi cos(2 pi x) y sin(pi y)
        // + v0 T0 pi x cos(pi x) cos(2 pi y)
        // + (2 T0 pi^2) / (Re Pr) cos(pi x) sin(pi y)
        EXPECT_NEAR(
            flux_integral,
            -u0 * T0 * pi * std::cos(2.0 * pi * x) * y * std::sin(pi * y) -
                v0 * T0 * pi * x * std::cos(pi * x) * std::cos(2.0 * pi * y) -
                diffusion * (2.0 * T0 * pi * pi) / (reynolds * prandtl) *
                    std::cos(pi * x) * std::sin(pi * y),
            1e-5);
        EXPECT_EQ(sd.lapl_T_flux_integral(src, i, j), flux_integral);
      }
    }
  }
}

TEST(flux_int_horiz, second_order_space_disc) {
  constexpr real T0 = 1.0;
  constexpr real u0 = 1.0;
  constexpr real v0 = 0.0;
  constexpr int cells_x = 512, cells_y = 512;
  constexpr std::pair<real, real> corner_1{0.0, 0.0}, corner_2{1.0, 1.0};
  // T = T0 cos(pi x) sin(pi y)
  // u = u0 y sin(pi x)
  // v = v0 x cos(pi y)
  Mesh src(construct_mesh_with_hbc(
      cells_x, cells_y, corner_1, corner_2, [&](real x, real y) {
        return triple{T0 * std::cos(pi * x) * std::sin(pi * y),
                      u0 * y * std::sin(pi * x), v0 * x * std::cos(pi * y)};
      }));

  constexpr real reynolds = 50.0;
  constexpr real prandtl = 0.7;
  constexpr real eckert = 0.1;
  constexpr real diffusion = 0.0;
  SecondOrderCentered sd(diffusion, reynolds, prandtl, eckert);
  for (int i = 0; i < src.cells_x(); i++) {
    for (int j = 0; j < src.cells_y(); j++) {
      const real x = src.median_x(i);
      const real y = src.median_y(j);
      const real flux_integral = sd.flux_integral(src, i, j);
      // u0 T0 pi cos(2 pi x) y sin(pi y)
      // + v0 T0 pi x cos(pi x) cos(2 pi y)
      // + (2 T0 pi^2) / (Re Pr) cos(pi x) sin(pi y)
      EXPECT_NEAR(
          flux_integral,
          -u0 * T0 * pi * std::cos(2.0 * pi * x) * y * std::sin(pi * y) -
              v0 * T0 * pi * x * std::cos(pi * x) * std::cos(2.0 * pi * y) -
              diffusion * (2.0 * T0 * pi * pi) / (reynolds * prandtl) *
                  std::cos(pi * x) * std::sin(pi * y),
          1e-4);
    }
  }
}

TEST(flux_int_vert, second_order_space_disc) {
  constexpr real T0 = 1.0;
  constexpr real u0 = 0.0;
  constexpr real v0 = 1.0;
  constexpr int cells_x = 512, cells_y = 512;
  constexpr std::pair<real, real> corner_1{0.0, 0.0}, corner_2{1.0, 1.0};
  // T = T0 cos(pi x) sin(pi y)
  // u = u0 y sin(pi x)
  // v = v0 x cos(pi y)
  Mesh src(construct_mesh_with_hbc(
      cells_x, cells_y, corner_1, corner_2, [&](real x, real y) {
        return triple{T0 * std::cos(pi * x) * std::sin(pi * y),
                      u0 * y * std::sin(pi * x), v0 * x * std::cos(pi * y)};
      }));

  constexpr real reynolds = 50.0;
  constexpr real prandtl = 0.7;
  constexpr real eckert = 0.1;
  constexpr real diffusion = 0.0;
  SecondOrderCentered sd(diffusion, reynolds, prandtl, eckert);
  for (int i = 0; i < src.cells_x(); i++) {
    for (int j = 0; j < src.cells_y(); j++) {
      const real x = src.median_x(i);
      const real y = src.median_y(j);
      const real flux_integral = sd.flux_integral(src, i, j);
      // u0 T0 pi cos(2 pi x) y sin(pi y)
      // + v0 T0 pi x cos(pi x) cos(2 pi y)
      // + (2 T0 pi^2) / (Re Pr) cos(pi x) sin(pi y)
      EXPECT_NEAR(
          flux_integral,
          -u0 * T0 * pi * std::cos(2.0 * pi * x) * y * std::sin(pi * y) -
              v0 * T0 * pi * x * std::cos(pi * x) * std::cos(2.0 * pi * y) -
              diffusion * (2.0 * T0 * pi * pi) / (reynolds * prandtl) *
                  std::cos(pi * x) * std::sin(pi * y),
          1e-4);
    }
  }
}

TEST(flux_int_complete, second_order_space_disc) {
  constexpr real T0 = 1.0;
  constexpr real u0 = 2.0;
  constexpr real v0 = 1.0;
  constexpr int cells_x = 512, cells_y = 512;
  constexpr std::pair<real, real> corner_1{0.0, 0.0}, corner_2{1.0, 1.0};
  // T = T0 cos(pi x) sin(pi y)
  // u = u0 y sin(pi x)
  // v = v0 x cos(pi y)
  Mesh src(construct_mesh_with_hbc(
      cells_x, cells_y, corner_1, corner_2, [&](real x, real y) {
        return triple{T0 * std::cos(pi * x) * std::sin(pi * y),
                      u0 * y * std::sin(pi * x), v0 * x * std::cos(pi * y)};
      }));

  constexpr real reynolds = 50.0;
  constexpr real prandtl = 0.7;
  constexpr real eckert = 0.1;
  constexpr real diffusion = 1.0;
  SecondOrderCentered sd(diffusion, reynolds, prandtl, eckert);
  for (int i = 0; i < src.cells_x(); i++) {
    for (int j = 0; j < src.cells_y(); j++) {
      const real x = src.median_x(i);
      const real y = src.median_y(j);
      const real flux_integral = sd.flux_integral(src, i, j);
      // u0 T0 pi cos(2 pi x) y sin(pi y)
      // + v0 T0 pi x cos(pi x) cos(2 pi y)
      // + (2 T0 pi^2) / (Re Pr) cos(pi x) sin(pi y)
      EXPECT_NEAR(
          flux_integral,
          -u0 * T0 * pi * std::cos(2.0 * pi * x) * y * std::sin(pi * y) -
              v0 * T0 * pi * x * std::cos(pi * x) * std::cos(2.0 * pi * y) -
              diffusion * (2.0 * T0 * pi * pi) / (reynolds * prandtl) *
                  std::cos(pi * x) * std::sin(pi * y),
          1e-3);
    }
  }
}

[[nodiscard]] Mesh construct_mesh_with_hbc(
    const int cells_x, const int cells_y, std::pair<real, real> corner_1,
    std::pair<real, real> corner_2,
    std::function<triple(real, real)> initial_conds) noexcept {
  Mesh src(corner_1, corner_2, cells_x, cells_y, initial_conds);
  // TODO: Make this less of a pain
  DirichletBC<Mesh::horiz_view>(src.ghostcells_top_Temp(),
                                src.bndrycells_top_Temp())
      .apply(0.0);
  DirichletBC<Mesh::horiz_view>(src.ghostcells_bottom_Temp(),
                                src.bndrycells_bottom_Temp())
      .apply(0.0);
  NeumannBC<Mesh::vert_view>(src.ghostcells_left_Temp(),
                             src.bndrycells_left_Temp())
      .apply(0.0);
  NeumannBC<Mesh::vert_view>(src.ghostcells_right_Temp(),
                             src.bndrycells_right_Temp())
      .apply(0.0);
  // The velocity bc isn't homogeneous Neumann or Dirichlet here
  // Physically the horizontal velocity in the top and bottom ghost cells is
  // degenerate
  // DirichletBC<Mesh::horiz_view>(src.ghostcells_top_vel_u(),
  //                               src.bndrycells_top_vel_u())
  //     .apply(0.0);
  // DirichletBC<Mesh::horiz_view>(src.ghostcells_bottom_vel_u(),
  //                               src.bndrycells_bottom_vel_u())
  //     .apply(0.0);
  DirichletBC<Mesh::vert_view>(src.ghostcells_left_vel_u(),
                               src.bndrycells_left_vel_u())
      .apply(0.0);
  DirichletBC<Mesh::vert_view>(src.ghostcells_right_vel_u(),
                               src.bndrycells_right_vel_u())
      .apply(0.0);
  NeumannBC<Mesh::horiz_view>(src.ghostcells_top_vel_v(),
                              src.bndrycells_top_vel_v())
      .apply(0.0);
  NeumannBC<Mesh::horiz_view>(src.ghostcells_bottom_vel_v(),
                              src.bndrycells_bottom_vel_v())
      .apply(0.0);
  // The velocity bc isn't homogeneous Neumann or Dirichlet here
  // This velocity component is also degenerate in these ghost cells
  // DirichletBC<Mesh::vert_view>(src.ghostcells_left_vel_v(),
  //                              src.bndrycells_left_vel_v())
  //     .apply(0.0);
  // DirichletBC<Mesh::vert_view>(src.ghostcells_right_vel_v(),
  //                              src.bndrycells_right_vel_v())
  //     .apply(0.0);
  return src;
}

TEST(boundary_cond, homogeneous) {
  // Verifies that the boundary conditions are only applied to the ghost
  // cells, and that the boundary value is 0.0
  constexpr int cells_x = 32, cells_y = 64;
  const std::pair<real, real> corner_1{0.0, 0.0}, corner_2{1.0, 1.0};

  std::random_device rd;
  std::mt19937_64 rng(rd());
  std::uniform_real_distribution<real> pdf(-100.0, 100.0);

  Mesh src(corner_1, corner_2, cells_x, cells_y, [&](real, real) {
    return triple{pdf(rng), pdf(rng), pdf(rng)};
  });
  const Mesh saved(src);
  // Verify src.Temp is the same as saved.Temp
  for (int i = 0; i < src.cells_x(); i++) {
    for (int j = 0; j < src.cells_y(); j++) {
      EXPECT_EQ(src.Temp(i, j), saved.Temp(i, j));
    }
  }
  DirichletBC top_bc(src.ghostcells_top_Temp(), src.bndrycells_top_Temp());
  top_bc.apply(0.0);
  // Check that every cell is untouched except for the top ghost cells
  for (int i = -1; i < src.cells_x() + 1; i++) {
    for (int j = -1; j < src.cells_y(); j++) {
      if (std::isnan(saved.Temp(i, j))) {
        EXPECT_TRUE(std::isnan(src.Temp(i, j)));
      } else {
        EXPECT_EQ(saved.Temp(i, j), src.Temp(i, j));
      }
    }
  }
  // The top corners should still be NaN
  EXPECT_TRUE(std::isnan(src.Temp(-1, src.cells_y())));
  EXPECT_TRUE(std::isnan(src.Temp(src.cells_x(), src.cells_y())));

  // Ensure that the boundary condition is met - since it's homogeneous it
  // should be exact
  for (int i = 0; i < src.cells_x(); i++) {
    EXPECT_FALSE(std::isnan(src.Temp(i, src.cells_y() - 1)));
    EXPECT_FALSE(std::isnan(src.Temp(i, src.cells_y())));
    EXPECT_EQ(src.Temp(i, src.cells_y() - 1), -src.Temp(i, src.cells_y()));
  }

  NeumannBC bottom_bc(src.ghostcells_bottom_Temp(),
                      src.bndrycells_bottom_Temp());
  bottom_bc.apply(0.0);
  // Check that every cell is untouched except for the top and bottom ghost
  // cells
  for (int i = -1; i < src.cells_x() + 1; i++) {
    for (int j = 0; j < src.cells_y(); j++) {
      if (std::isnan(saved.Temp(i, j))) {
        EXPECT_TRUE(std::isnan(src.Temp(i, j)));
      } else {
        EXPECT_EQ(saved.Temp(i, j), src.Temp(i, j));
      }
    }
  }
  // The corners should still be NaN
  EXPECT_TRUE(std::isnan(src.Temp(-1, -1)));
  EXPECT_TRUE(std::isnan(src.Temp(src.cells_x(), -1)));
  EXPECT_TRUE(std::isnan(src.Temp(-1, src.cells_y())));
  EXPECT_TRUE(std::isnan(src.Temp(src.cells_x(), src.cells_y())));

  // Ensure that the boundary condition is met - since it's homogeneous it
  // should be exact
  for (int i = 0; i < src.cells_x(); i++) {
    EXPECT_FALSE(std::isnan(src.Temp(i, 0)));
    EXPECT_FALSE(std::isnan(src.Temp(i, -1)));
    EXPECT_EQ(src.Temp(i, 0), src.Temp(i, -1));
  }

  NeumannBC right_bc(src.ghostcells_right_Temp(), src.bndrycells_right_Temp());
  right_bc.apply(0.0);
  // Check that every cell is untouched except for the right, top, and bottom
  // ghost cells
  for (int i = -1; i < src.cells_x(); i++) {
    for (int j = 0; j < src.cells_y(); j++) {
      if (std::isnan(saved.Temp(i, j))) {
        EXPECT_TRUE(std::isnan(src.Temp(i, j)));
      } else {
        EXPECT_EQ(saved.Temp(i, j), src.Temp(i, j));
      }
    }
  }
  // The corners should still be NaN
  EXPECT_TRUE(std::isnan(src.Temp(-1, -1)));
  EXPECT_TRUE(std::isnan(src.Temp(src.cells_x(), -1)));
  EXPECT_TRUE(std::isnan(src.Temp(-1, src.cells_y())));
  EXPECT_TRUE(std::isnan(src.Temp(src.cells_x(), src.cells_y())));

  // Ensure that the boundary condition is met - since it's homogeneous it
  // should be exact
  for (int j = 0; j < src.cells_y(); j++) {
    EXPECT_FALSE(std::isnan(src.Temp(src.cells_x() - 1, j)));
    EXPECT_FALSE(std::isnan(src.Temp(src.cells_x(), j)));
    EXPECT_EQ(src.Temp(src.cells_x() - 1, j), src.Temp(src.cells_x(), j));
  }

  DirichletBC left_bc(src.ghostcells_left_Temp(), src.bndrycells_left_Temp());
  left_bc.apply(0.0);
  // Check that every interior cell is untouched
  for (int i = 0; i < src.cells_x(); i++) {
    for (int j = 0; j < src.cells_y(); j++) {
      if (std::isnan(saved.Temp(i, j))) {
        EXPECT_TRUE(std::isnan(src.Temp(i, j)));
      } else {
        EXPECT_EQ(saved.Temp(i, j), src.Temp(i, j));
      }
    }
  }
  // The corners should still be NaN
  EXPECT_TRUE(std::isnan(src.Temp(-1, -1)));
  EXPECT_TRUE(std::isnan(src.Temp(src.cells_x(), -1)));
  EXPECT_TRUE(std::isnan(src.Temp(-1, src.cells_y())));
  EXPECT_TRUE(std::isnan(src.Temp(src.cells_x(), src.cells_y())));

  // Ensure that the boundary condition is met - since it's homogeneous it
  // should be exact
  for (int j = 0; j < src.cells_y(); j++) {
    EXPECT_FALSE(std::isnan(src.Temp(0, j)));
    EXPECT_FALSE(std::isnan(src.Temp(-1, j)));
    EXPECT_EQ(src.Temp(-1, j), -src.Temp(0, j));
  }
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}


#include <cmath>
#include <gtest/gtest.h>
#include <random>

#include "boundary_conds.hpp"
#include "constants.hpp"
#include "mesh.hpp"

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

TEST(boundary_cond, homogeneous) {
  // Verifies that the boundary conditions are only applied to the ghost cells,
  // and that the boundary value is 0.0
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


#ifndef _SPACE_DISC_HPP_
#define _SPACE_DISC_HPP_

#include "constants.hpp"
#include "mesh.hpp"

#include <memory>

// Use the curiously recurring template parameter to provide the discretization
// approximations to the derivatives
template <typename SpaceDisc_> class EnergyAssembly {
public:
  using SpaceDisc = SpaceDisc_;

  using vert_view = Mesh::vert_view;
  using horiz_view = Mesh::horiz_view;

  [[nodiscard]] real flux_integral(const Mesh &mesh, int i, int j) const
      noexcept {
    const real x_deriv =
        (static_cast<const SpaceDisc *>(this)->uT_x_flux(mesh, i, j) -
         static_cast<const SpaceDisc *>(this)->uT_x_flux(mesh, i - 1, j)) /
        mesh.dx();
    const real y_deriv =
        (static_cast<const SpaceDisc *>(this)->vT_y_flux(mesh, i, j) -
         static_cast<const SpaceDisc *>(this)->vT_y_flux(mesh, i, j - 1)) /
        mesh.dy();

    return (-x_deriv - y_deriv) + diffusion_ * lapl_T_flux_integral(mesh, i, j);
  }

  [[nodiscard]] real lapl_T_flux_integral(const Mesh &mesh, int i, int j) const
      noexcept {
    const real x2_deriv =
        (static_cast<const SpaceDisc *>(this)->dx_flux(mesh, i, j) -
         static_cast<const SpaceDisc *>(this)->dx_flux(mesh, i - 1, j)) /
        mesh.dx();
    const real y2_deriv =
        (static_cast<const SpaceDisc *>(this)->dy_flux(mesh, i, j) -
         static_cast<const SpaceDisc *>(this)->dy_flux(mesh, i, j - 1)) /
        mesh.dy();
    return (x2_deriv + y2_deriv) * inv_rp_;
  }

  void flux_assembly(const Mesh &initial, const Mesh &current, Mesh &next,
                     const real dt) const noexcept {
    for (int i = 0; i < initial.cells_x(); i++) {
      for (int j = 0; j < initial.cells_y(); j++) {
        next.Temp(i, j) =
            initial.Temp(i, j) +
            dt * (flux_integral(current, i, j) +
                  static_cast<SpaceDisc *>(this)->source_fd(current, i, j));
      }
    }
  }

  constexpr EnergyAssembly(const real diffusion, const real reynolds,
                           const real prandtl, const real eckert)
      : diffusion_(diffusion), reynolds_(reynolds), prandtl_(prandtl),
        eckert_(eckert), inv_rp_(1.0 / (reynolds * prandtl)) {}

protected:
  real diffusion_;
  real reynolds_, prandtl_, eckert_;
  real inv_rp_;
};

class[[nodiscard]] SecondOrderCentered
    : public EnergyAssembly<SecondOrderCentered> {
public:
  using Assembly = EnergyAssembly<SecondOrderCentered>;

  // Terms for implicit euler
  [[nodiscard]] constexpr real Dx_p1(const Mesh &mesh, int i, int j)
      const noexcept {
    if (i < mesh.cells_x()) {
      return (mesh.vel_u(i + 1, j) / 2.0 - this->inv_rp_ / mesh.dx()) /
             mesh.dx();
    } else {
      return 0.0;
    }
  }

  [[nodiscard]] constexpr real Dx_0(const Mesh &mesh, int i, int j)
      const noexcept {
    return 2.0 * this->inv_rp_ / (mesh.dx() * mesh.dx());
  }

  [[nodiscard]] constexpr real Dx_m1(const Mesh &mesh, int i, int j)
      const noexcept {
    if (i >= 0) {
      return (-mesh.vel_u(i - 1, j) / 2.0 - this->inv_rp_ / mesh.dx()) /
             mesh.dx();
    } else {
      return 0.0;
    }
  }

  [[nodiscard]] constexpr real Dy_p1(const Mesh &mesh, int i, int j)
      const noexcept {
    if (j < mesh.cells_y()) {
      return (mesh.vel_v(i, j + 1) / 2.0 - this->inv_rp_ / mesh.dy()) /
             mesh.dy();
    } else {
      return 0.0;
    }
  }

  [[nodiscard]] constexpr real Dy_0(const Mesh &mesh, int i, int j)
      const noexcept {
    return 2.0 * this->inv_rp_ / (mesh.dy() * mesh.dy());
  }

  [[nodiscard]] constexpr real Dy_m1(const Mesh &mesh, int i, int j)
      const noexcept {
    if (j >= 0) {
      return (-mesh.vel_v(i, j - 1) / 2.0 - this->inv_rp_ / mesh.dy()) /
             mesh.dy();
    } else {
      return 0.0;
    }
  }

  // Centered FV approximation to (u T)_{i+1/2, j}
  [[nodiscard]] constexpr real uT_x_flux(const Mesh &mesh, const int i,
                                         const int j) const noexcept {
    return (mesh.Temp(i, j) * mesh.vel_u(i, j) +
            mesh.Temp(i + 1, j) * mesh.vel_u(i + 1, j)) /
           2.0;
  }

  // Centered FV approximation to T_{i, j+1/2}
  [[nodiscard]] constexpr real vT_y_flux(const Mesh &mesh, const int i,
                                         const int j) const noexcept {
    return (mesh.Temp(i, j) * mesh.vel_v(i, j) +
            mesh.Temp(i, j + 1) * mesh.vel_v(i, j + 1)) /
           2.0;
  }

  // Centered FV approximation to dT/dx_{i+1/2, j}
  [[nodiscard]] constexpr real dx_flux(const Mesh &mesh, const int i,
                                       const int j) const noexcept {
    return (mesh.Temp(i + 1, j) - mesh.Temp(i, j)) / mesh.dx();
  }

  // Centered FV approximation to dT/dy_{i, j+1/2}
  [[nodiscard]] constexpr real dy_flux(const Mesh &mesh, const int i,
                                       const int j) const noexcept {
    return (mesh.Temp(i, j + 1) - mesh.Temp(i, j)) / mesh.dy();
  }

  // Uses the finite difference (FD) approximations to the velocity derivatives
  // to approximate the source term
  [[nodiscard]] constexpr real source_fd(const Mesh &mesh, const int i,
                                         const int j) const noexcept {
    const real u_dx = du_dx_fd(mesh, i, j);
    const real v_dy = dv_dy_fd(mesh, i, j);
    const real u_dy = du_dy_fd(mesh, i, j);
    const real v_dx = dv_dx_fd(mesh, i, j);

    const real cross_term = u_dy + v_dx;
    return eckert_ / reynolds_ *
           (2.0 * (u_dx * u_dx + v_dy * v_dy) + cross_term * cross_term);
  }

  // Centered FD approximation to du/dx_{i, j}
  [[nodiscard]] constexpr real du_dx_fd(const Mesh &mesh, const int i,
                                        const int j) const noexcept {
    return (mesh.vel_u(i + 1, j) - mesh.vel_u(i - 1, j)) / (2.0 * mesh.dx());
  }

  // Centered FD approximation to du/dy_{i, j}
  [[nodiscard]] constexpr real du_dy_fd(const Mesh &mesh, const int i,
                                        const int j) const noexcept {
    return (mesh.vel_u(i, j + 1) - mesh.vel_u(i, j - 1)) / (2.0 * mesh.dy());
  }

  // Centered FD approximation to dv/dx_{i, j}
  [[nodiscard]] constexpr real dv_dx_fd(const Mesh &mesh, const int i,
                                        const int j) const noexcept {
    return (mesh.vel_v(i + 1, j) - mesh.vel_v(i - 1, j)) / (2.0 * mesh.dx());
  }

  // Centered FD approximation to dv/dy_{i, j}
  [[nodiscard]] constexpr real dv_dy_fd(const Mesh &mesh, const int i,
                                        const int j) const noexcept {
    return (mesh.vel_v(i, j + 1) - mesh.vel_v(i, j - 1)) / (2.0 * mesh.dy());
  }

  constexpr SecondOrderCentered(const real diffusion, const real reynolds,
                                const real prandtl, const real eckert) noexcept
      : Assembly(diffusion, reynolds, prandtl, eckert) {}
};

#endif // _SPACE_DISC_HPP_

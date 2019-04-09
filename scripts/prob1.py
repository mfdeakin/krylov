# When running with address sanitizer and built with Clang, this needs the following environment variable:
# LD_PRELOAD=${PATH_TO_LIB}/libclang_rt.asan.so
# When built with gcc it needs some equivalent library
from krylov import SecondOrderCentered, Mesh, BCType
from krylov import (ImplicitEulerLUSolver as IELU,
                    ImplicitEulerAFSolver as IEAF,
                    ImplicitEulerGMRESTDPSolver10 as IEGMRES10)
from krylov import BoundaryConditionHoriz as BCH
from krylov import BoundaryConditionVert as BCV

from numpy import set_printoptions
from numpy import sin, cos, pi, nan, sum, sqrt, array, real

from scipy.linalg import eig

from matplotlib.pyplot import figure, contour, show, clabel, title, pcolor, colorbar, semilogy, legend
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

import time

set_printoptions(precision=4, suppress=True, sign=' ', threshold=2501, linewidth=nan,
                 formatter={"float": lambda f: "{:8.5}".format(f)})

def plot_3d(x, y, a, name):
    fig = figure()
    ax = fig.gca(projection="3d")
    s = ax.plot_surface(x, y, a, cmap=cm.gist_heat)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    colorbar(s)
    title(name)

def plot_Temp_mesh(m, name):
    Temp = m.Temp_array()
    # Use a weighted average to set the corners to something other than NaN
    Temp[0, 0] = (Temp[1, 1] + 2.0 * Temp[0, 1] + 2.0 * Temp[1, 0]) / 5.0
    Temp[-1, 0] = (Temp[-2, 1] + 2.0 * Temp[-1, 1] + 2.0 * Temp[-2, 0]) / 5.0
    Temp[0, -1] = (Temp[1, -2] + 2.0 * Temp[0, -2] + 2.0 * Temp[1, -1]) / 5.0
    Temp[-1, -1] = (Temp[-2, -2] + 2.0 * Temp[-1, -2] + 2.0 * Temp[-1, -2]) / 5.0
    plot_3d(m.grid_x(), m.grid_y(),
            Temp, name)

def plot_vel_u_mesh(m, name):
    vel_u = m.vel_u_array()
    # Use a weighted average to set the corners to something other than NaN
    vel_u[0, 0] = (vel_u[1, 1] + 2.0 * vel_u[0, 1] + 2.0 * vel_u[1, 0]) / 5.0
    vel_u[-1, 0] = (vel_u[-2, 1] + 2.0 * vel_u[-1, 1] + 2.0 * vel_u[-2, 0]) / 5.0
    vel_u[0, -1] = (vel_u[1, -2] + 2.0 * vel_u[0, -2] + 2.0 * vel_u[1, -1]) / 5.0
    vel_u[-1, -1] = (vel_u[-2, -2] + 2.0 * vel_u[-1, -2] + 2.0 * vel_u[-1, -2]) / 5.0
    plot_3d(m.grid_x(), m.grid_y(),
            vel_u, name)

def plot_vel_v_mesh(m, name):
    vel_v = m.vel_v_array()
    # Use a weighted average to set the corners to something other than NaN
    vel_v[0, 0] = (vel_v[1, 1] + 2.0 * vel_v[0, 1] + 2.0 * vel_v[1, 0]) / 5.0
    vel_v[-1, 0] = (vel_v[-2, 1] + 2.0 * vel_v[-1, 1] + 2.0 * vel_v[-2, 0]) / 5.0
    vel_v[0, -1] = (vel_v[1, -2] + 2.0 * vel_v[0, -2] + 2.0 * vel_v[1, -1]) / 5.0
    vel_v[-1, -1] = (vel_v[-2, -2] + 2.0 * vel_v[-1, -2] + 2.0 * vel_v[-1, -2]) / 5.0
    plot_3d(m.grid_x(), m.grid_y(),
            vel_v, name)

def def_configuration(timedisc):
    diffusion = 1.0
    reynolds = 25.0
    prandtl = 0.7
    eckert = 0.1
    avg_u = 3.0
    return timedisc((0.0, 0.0), (40.0, 1.0), 38, 18,
                    lambda x, y: (0.0,
                                  6.0 * avg_u * y * (1.0 - y), 0.0),
                    SecondOrderCentered(diffusion, reynolds, prandtl, eckert),
                    (BCType.dirichlet, lambda x, y, t: 0.0), # Temperature bottom
                    (BCType.dirichlet, lambda x, y, t: 1.0), # Temperature top
                    (BCType.dirichlet, lambda x, y, t: y),   # Temperature left
                    # (BCType.dirichlet, lambda x, y, t: y + (0.75 * prandtl * eckert * avg_u * avg_u
                    #                                         * (1.0 - (1.0 - 2.0 * y) ** 4))), # Temperature right
                    (BCType.neumann, lambda x, y, t: 0.0), # Temperature right
                    (BCType.dirichlet, lambda x, y, t: 0.0), # Vel u bottom
                    (BCType.dirichlet, lambda x, y, t: 0.0), # Vel u top
                    (BCType.neumann, lambda x, y, t: 0.0), # Vel u left
                    (BCType.neumann, lambda x, y, t: 0.0), # Vel u right
                    (BCType.dirichlet, lambda x, y, t: 0.0), # Vel v bottom
                    (BCType.dirichlet, lambda x, y, t: 0.0), # Vel v top
                    (BCType.dirichlet, lambda x, y, t: 0.0), # Vel v left
                    (BCType.dirichlet, lambda x, y, t: 0.0)) # Vel v right

solver = def_configuration(IEGMRES10)
plot_Temp_mesh(solver.mesh(), "Initial Temperature Mesh")
plot_vel_u_mesh(solver.mesh(), "Velocity u Mesh")
max_delta = float('inf')
i = 0
while max_delta > 1e-9:
    print("Timestep {}, delta: {}".format(i, max_delta))
    max_delta = solver.timestep(0.25)
    i += 1
    if i < 10:
        plot_Temp_mesh(solver.mesh(), "Temperature Mesh at t={:.2} ({} iterations)".format(solver.cur_time(), i))

plot_Temp_mesh(solver.mesh(), "Temperature Mesh at t={:.2} ({} iterations)".format(solver.cur_time(), i))

# print(solver.mesh().cells_x(), solver.mesh().cells_y())
# solver_lu = def_configuration(IELU)
# sys = solver_lu.system().to_array()
# sol = solver.solution().to_array()
# result = solver.delta().to_array()

# print(sys)
# print()
# print(sol)
# print()
# print(result)
# print()
# print(sol - sys.dot(result))
# print()
# e, _ = eig(sys)
# print("Eigenvalues:", e)
# print()
# print("Eigenvalue Real components:", real(e))
# solver = def_configuration(IEGMRES10)
# solver.timestep(0.25)
# print("Comparison")
# print("LU  :", result)
# result = solver.delta().to_array()
# print("GMRES:", result)
# print()
# print("GMRES Residual:", sol - sys.dot(result))

# solver = def_configuration(IEAF)
# sys_dy = solver.assemble_system_dy(0.25, 0)
# print("dy system:")
# print(sys_dy.to_array())
# sys_dx = solver.assemble_system_dx(0.25, 0)
# print("dx system:")
# print(sys_dx.to_array())

show()

from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import fem, mesh, default_scalar_type, log
from dolfinx.fem import Constant, Function, Expression
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from dolfinx.io import VTXWriter
import ufl
from datetime import datetime
from hilliard_models import Cahn_Hillard_Plane_Strain
import pyvista

import numpy as np

pyvista.set_jupyter_backend("client")
## Define temporal parameters


problemName = "Canh Hillard Plane With Mech"

# Square edge length
L0 = 0.8  # 800 nm box, after Di Leo et al. (2014)

# Number of elements along each side
N = 100

# Create square mesh
domain = mesh.create_rectangle(MPI.COMM_WORLD, [(0, 0), (L0, L0)], [N, N])

t = 0.0  # initialization of time
Ttot = 2000  # total simulation time
dt = 0.05  # Initial time step size, here we will use adaptive time-stepping

dk = Constant(domain, dt)

hillard_problem = Cahn_Hillard_Plane_Strain(domain)
hillard_problem.Kinematics()
hillard_problem.WeakForms(dk)


# Add Boundary Conditions:


def bottom(x):
    return np.isclose(x[1], 0)


def right(x):
    return np.isclose(x[0], 0)


fdim = domain.topology.dim - 1
left_facets = mesh.locate_entities_boundary(domain, fdim, right)
bottom_facets = mesh.locate_entities_boundary(domain, fdim, bottom)

marked_facets = np.hstack([bottom_facets, left_facets])
marked_values = np.hstack(
    [np.full_like(bottom_facets, 1), np.full_like(left_facets, 2)]
)
sorted_facets = np.argsort(marked_facets)

facet_tag = mesh.meshtags(
    domain, fdim, marked_facets[sorted_facets], marked_values[sorted_facets]
)


# Just fix the sides to make sure they don't move

u_bc = np.array((0), dtype=default_scalar_type)


left_dofs = fem.locate_dofs_topological(
    hillard_problem.ME.sub(0).sub(0), facet_tag.dim, facet_tag.find(2)
)  # we don't want it to move in the x direction
bottom_dofs = fem.locate_dofs_topological(
    hillard_problem.ME.sub(0).sub(1), facet_tag.dim, facet_tag.find(1)
)  # we don't want it to move in the y direction
bcs = [
    fem.dirichletbc(u_bc, left_dofs, hillard_problem.ME.sub(0).sub(0)),
    fem.dirichletbc(u_bc, bottom_dofs, hillard_problem.ME.sub(0).sub(1)),
]


# # SETUP NONLINEAR PROBLEM
U1 = ufl.VectorElement("Lagrange", domain.ufl_cell(), 1)
V2 = fem.FunctionSpace(domain, U1)  # Vector function space
V1 = fem.FunctionSpace(domain, hillard_problem.P1)  # Scalar function space

u_vis = Function(V2)
u_vis.name = "u"
u_expr = Expression(hillard_problem.u, V2.element.interpolation_points())


mu_vis = Function(V1)
mu_vis.name = "mu"
mu_expr = Expression(hillard_problem.mu, V1.element.interpolation_points())

c_vis = Function(V1)
c_vis.name = "c"
c_expr = Expression(hillard_problem.c, V1.element.interpolation_points())


T = hillard_problem.T
T2D = ufl.as_tensor([[T[0, 0], T[0, 1]], [T[1, 0], T[1, 1]]])

sigma1, sigma2, vec1, vec2 = hillard_problem.eigs(T2D)

sigma1_vis = Function(V1)
sigma1_vis.name = "sigma1"
sigma1_expr = Expression(
    sigma1 * hillard_problem.Gshear, V1.element.interpolation_points()
)

sigma2_vis = Function(V1)
sigma2_vis.name = "sigma2"
sigma2_expr = Expression(
    sigma2 * hillard_problem.Gshear, V1.element.interpolation_points()
)


# vtk2 = VTXWriter(domain.comm,"results/"+problemName+"displacement.bp", [u_vis], engine="BP4" )

vtk = VTXWriter(
    domain.comm,
    "results/" + problemName + "scalrValues.bp",
    [u_vis, mu_vis, c_vis, sigma1_vis, sigma2_vis],
    engine="BP4",
)

files = [vtk]


def interp_and_save(t, files: list[VTXWriter]):
    u_vis.interpolate(u_expr)
    mu_vis.interpolate(mu_expr)
    c_vis.interpolate(c_expr)
    sigma2_vis.interpolate(sigma2_expr)
    sigma1_vis.interpolate(sigma1_expr)

    for file in files:
        file.write(t)


step = "Swell"
jit_options = {"cffi_extra_compile_args": ["-O3", "-ffast-math"]}

problem = NonlinearProblem(
    hillard_problem.Res, hillard_problem.w, bcs, hillard_problem.a
)


solver = NewtonSolver(MPI.COMM_WORLD, problem)
solver.convergence_criterion = "incremental"
solver.rtol = 1e-8
solver.atol = 1e-8
solver.max_it = 100
solver.report = True

solver.error_on_nonconvergence = False

ksp = solver.krylov_solver
opts = PETSc.Options()
option_prefix = ksp.getOptionsPrefix()

opts[f"{option_prefix}ksp_max_it"] = 60
opts[f"{option_prefix}ksp_type"] = "preonly"
opts[f"{option_prefix}pc_type"] = "lu"

# opts[f"{option_prefix}mg_levels_ksp_type"] = "chebyshev"
# opts[f"{option_prefix}mg_levels_pc_type"] = "jacobi"


# opts[f"{option_prefix}mg_levels_esteig_ksp_type"] = "cg"
# opts[f"{option_prefix}mg_levels_ksp_chebyshev_esteig_steps"] = 50

# opts[f"{option_prefix}pc_hypre_type"] = "boomeramg"
# opts[f"{option_prefix}malloc_debug"] = "True"
opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
# pts[f"{option_prefix}monitor_convergence"] = True


# ksp.parameters["monitor_convergence"] = True
ksp.setFromOptions()
startTime = datetime.now()
if domain.comm.rank == 0:
    print("------------------------------------")
    print("Simulation Start")
    print("------------------------------------")

step = "Evolve"

#  log.set_log_level(log.LogLevel.INFO)
interp_and_save(t, files)
ii = 0
bisection_count = 0
while t < Ttot:
    # increment time
    t += float(dk)
    # increment counter
    ii += 1

    # Solve the problem

    (iter, converged) = solver.solve(hillard_problem.w)

    if converged:
        hillard_problem.w.x.scatter_forward()
        hillard_problem.w_old_2.x.array[:] = hillard_problem.w_old.x.array
        hillard_problem.w_old.x.array[:] = hillard_problem.w.x.array

        interp_and_save(t, files)
        if ii % 1 == 0 and domain.comm.rank == 0:
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print("Step: {} |   Increment: {} | Iterations: {}".format(step, ii, iter))
            print("Simulation Time: {} s | dt: {} s".format(round(t, 2), round(dt, 3)))
            print()

        if iter <= 3:
            dt = 1.5 * dt

            if dt > 1:
                dt = 1
            dk.value = dt

        # If the newton solver takes 5 or more iterations,
        # decrease the time step by a factor of 2:
        elif iter >= 6:
            dt = dt / 2
            dk.value = dt

        # Reset Biseciton Counter
        bisection_count = 0

    else:
        # Break the loop if solver fails too many times
        bisection_count += 1

        if bisection_count > 5:
            print("Error: Too many bisections")
            break

        print("Error Halfing Time Step")
        t = t - float(dk)
        dt = dt / 2
        dk.value = dt
        print(f"New Time Step: {dt}")
        hillard_problem.w.x.array[:] = hillard_problem.w_old.x.array
        hillard_problem.w_old.x.array[:] = hillard_problem.w_old_2.x.array


# End Analysis
vtk.close()
endTime = datetime.now()
if domain.comm.rank == 0:
    print("------------------------------------")
    print("Simulation End")
    print("------------------------------------")
    print("Total Time: {}".format(endTime - startTime))
    print("------------------------------------")

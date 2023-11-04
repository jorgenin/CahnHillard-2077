# %% [markdown]
# # Code for Cahn-Hilliard phase separation without mechanical coupling.
#
# ## 2D phase separation study.
#
#
# Degrees of freedom:
#   - scalar chemical potential: we use normalized  mu = mu/RT
#   - species concentration:  we use normalized  c= Omega*cmat
#
# ### Units:
# - Length: um
# - Mass: kg
# - Time: s
# - Amount of substance: pmol
# - Temperature: K
# - Mass density: kg/um^3
# - Force: uN
# - Stress: MPa
# - Energy: pJ
# - Species concentration: pmol/um^3
# - Chemical potential: pJ/pmol
# - Molar volume: um^3/pmol
# - Species diffusivity: um^2/s
# - Boltzmann Constant: 1.38E-11 pJ/K
# - Gas constant: 8.314  pJ/(pmol K)
#
# ### By
#   Eric Stewart      and      Lallit Anand
# ericstew@mit.edu            anand@mit.edu
#
# October 2023
#
# Modified for FenicsX by Jorge Nin
# jorgenin@mit.edu

# %%
import numpy as np


from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import fem, mesh, io, plot, log
from dolfinx.fem import Constant, dirichletbc, Function, FunctionSpace, Expression
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from dolfinx.io import VTXWriter
import ufl
from ufl import (
    TestFunction,
    TrialFunction,
    Identity,
    grad,
    det,
    div,
    dev,
    inv,
    tr,
    sqrt,
    conditional,
    gt,
    dx,
    inner,
    derivative,
    dot,
    ln,
    split,
    tanh,
    as_tensor,
    as_vector,
    ge,
)
from datetime import datetime
from dolfinx.plot import vtk_mesh
from hilliard_models import Cahn_Hillard_3D_no_mech


import random

problemName = "Canh Hillard 3D"

# Square edge length
L0 = 0.8  # 800 nm box, after Di Leo et al. (2014)

# Number of elements along each side
N = 50

# Create square mesh
domain = mesh.create_box(
    MPI.COMM_WORLD,
    [(0, 0, 0), (L0, L0, L0)],
    [N, N, N],
    cell_type=mesh.CellType.hexahedron,
)

# %%
t = 0.0  # initialization of time
Ttot = 4  # total simulation time
dt = 0.01  # Initial time step size, here we will use adaptive time-stepping

# %%
dk = Constant(domain, dt)

hillard_problem = Cahn_Hillard_3D_no_mech(domain)
hillard_problem.Kinematics()
hillard_problem.WeakForms(dk)

U1 = ufl.VectorElement("Lagrange", domain.ufl_cell(), 1)
V2 = fem.FunctionSpace(domain, U1)  # Vector function space
V1 = fem.FunctionSpace(domain, hillard_problem.P1)  # Scalar function space

mu_vis = Function(V1)
mu_vis.name = "mu"
mu_expr = Expression(hillard_problem.mu, V1.element.interpolation_points())

c_vis = Function(V1)
c_vis.name = "c"
c_expr = Expression(hillard_problem.c, V1.element.interpolation_points())

vtk = VTXWriter(
    domain.comm, "results/" + problemName + ".bp", [mu_vis, c_vis], engine="BP4"
)


def interp_and_save(t, file):
    mu_vis.interpolate(mu_expr)
    c_vis.interpolate(c_expr)

    file.write(t)


# Nothing! Just let the system evolve on its own.
bcs = []


import os

step = "Swell"
jit_options = {"cffi_extra_compile_args": ["-O3", "-ffast-math"]}

problem = NonlinearProblem(
    hillard_problem.Res,
    hillard_problem.w,
    bcs,
    hillard_problem.a,
    jit_options=jit_options,
)


solver = NewtonSolver(MPI.COMM_WORLD, problem)
solver.convergence_criterion = "incremental"
solver.rtol = 1e-8
solver.atol = 1e-8
solver.max_it = 50
solver.report = True


ksp = solver.krylov_solver
opts = PETSc.Options()
option_prefix = ksp.getOptionsPrefix()
opts[f"{option_prefix}ksp_max_it"] = 30
opts[f"{option_prefix}ksp_type"] = "gmres"
opts[f"{option_prefix}pc_type"] = "ksp"
ksp.setFromOptions()

startTime = datetime.now()
print("------------------------------------")
print("Simulation Start")
print("------------------------------------")

step = "Evolve"

# if os.path.exists("results/"+problemName+".bp"):
#    os.remove("results/"+problemName+".xdmf")
#    os.remove("results/"+problemName+".h5")

# vtk.write_mesh(domain)
t = 0.0

interp_and_save(t, vtk)
ii = 0
bisection_count = 0
lasttimestep = hillard_problem.w_old_2.x.array.copy()
while t < Ttot:
    # increment time
    t += float(dk)
    # increment counter
    ii += 1

    # Solve the problem
    try:
        (iter, converged) = solver.solve(hillard_problem.w)

        hillard_problem.w.x.scatter_forward()

        lasttimestep[:] = hillard_problem.w_old_2.x.array
        hillard_problem.w_old_2.x.array[:] = hillard_problem.w_old.x.array
        hillard_problem.w_old.x.array[:] = hillard_problem.w.x.array

        interp_and_save(t, vtk)
        if ii % 1 == 0 and domain.comm.rank == 0:
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print("Step: {} |   Increment: {} | Iterations: {}".format(step, ii, iter))
            print("Simulation Time: {} s | dt: {} s".format(round(t, 2), round(dt, 3)))
            print()

        if iter <= 3:
            dt = 1.5 * dt
            dk.value = dt
        # If the newton solver takes 5 or more iterations,
        # decrease the time step by a factor of 2:
        elif iter >= 5:
            dt = dt / 2
            dk.value = dt

        # Reset Biseciton Counter
        bisection_count = 0

    except:  # Break the loop if solver fails
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
        hillard_problem.w_old_2.x.array[:] = lasttimestep


# End Analysis
vtk.close()
endTime = datetime.now()
print("------------------------------------")
print("Simulation End")
print("------------------------------------")
print("Total Time: {}".format(endTime - startTime))
print("------------------------------------")

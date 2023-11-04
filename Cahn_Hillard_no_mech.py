from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import fem, mesh
from dolfinx.fem import Constant, Function, Expression
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from dolfinx.io import VTXWriter
import ufl
from datetime import datetime
from hilliard_models import Cahn_Hillard_Plane_Strain_no_mech
import pyvista

pyvista.set_jupyter_backend("client")
## Define temporal parameters


problemName = "Canh Hillard Plane Strain No Mech"

# Square edge length
L0 = 0.8  # 800 nm box, after Di Leo et al. (2014)

# Number of elements along each side
N = 100

# Create square mesh
domain = mesh.create_rectangle(MPI.COMM_WORLD, [(0, 0), (L0, L0)], [N, N])

t = 0.0  # initialization of time
Ttot = 2000  # total simulation time
dt = 0.01  # Initial time step size, here we will use adaptive time-stepping

dk = Constant(domain, dt)

hillard_problem = Cahn_Hillard_Plane_Strain_no_mech(domain)
hillard_problem.Kinematics()
hillard_problem.WeakForms(dk)


# Nothing! Just let the system evolve on its own.
bcs = []


# # SETUP NONLINEAR PROBLEM
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
opts[f"{option_prefix}ksp_type"] = "none"
opts[f"{option_prefix}pc_type"] = "lu"
opts[f"{option_prefix}malloc_debug"] = "True"
opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"

ksp.setFromOptions()

startTime = datetime.now()
print("------------------------------------")
print("Simulation Start")
print("------------------------------------")

step = "Evolve"

# if os.path.exists("results/"+problemName+".bp"):
#    os.remove("results/"+problemName+".xdmf")
#    os.remove("results/"+problemName+".h5")

interp_and_save(t, vtk)
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

        interp_and_save(t, vtk)
        if ii % 1 == 0:
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


# End Analysis
vtk.close()
endTime = datetime.now()
print("------------------------------------")
print("Simulation End")
print("------------------------------------")
print("Total Time: {}".format(endTime - startTime))
print("------------------------------------")

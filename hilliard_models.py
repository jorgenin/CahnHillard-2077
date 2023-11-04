from __future__ import annotations
from mpi4py import MPI
from typing import Callable

from ufl import (
    TestFunctions,
    TrialFunction,
    Identity,
    as_tensor,
    as_vector,
    grad,
    det,
    inv,
    tr,
    sqrt,
    conditional,
    gt,
    inner,
    derivative,
    dot,
    ln,
    split,
    as_tensor,
    as_vector,
    ge,
    eq,
    SpatialCoordinate,
)
import ufl
from dolfinx.fem import Constant
from dolfinx.fem import (
    Constant,
    Function,
    FunctionSpace,
    Expression,
)

import numpy as np


class Cahn_Hillard_axi_symmetric:
    def __init__(self, domain, **kwargs):
        """
        Initializes a Cahn-Hilliard model with the given domain.

        Args:
            domain (dolfin.Mesh): The mesh domain for the model.
            **kwargs: Additional keyword arguments to be passed to the method.
        """

    def __init__(
        self,
        domain,
        init_func: Callable[[Cahn_Hillard_axi_symmetric], None] = None,
        **kwargs,
    ):
        # Create Contants

        self.Omega = Constant(domain, 4.05)  # Molar volume, um^3/pmol
        self.D = Constant(domain, 1e-2)  # Diffusivity, um^2/s
        self.chi = Constant(domain, 3.0)  # Phase parameter, (-)
        self.cMax = Constant(domain, 2.29e-2)  # Saturation concentration, pmol/um^3
        self.lam = Constant(domain, 5.5749e-1)  # Interface parameter, (pJ/pmol) um^2
        self.theta0 = Constant(domain, 298.0)  # Reference temperature, K
        self.R_gas = Constant(domain, 8.3145)  # Gas constant, pJ/(pmol K)

        self.Gshear = Constant(domain, 4980.0)  # Shear modulus, MPa
        self.K = Constant(domain, 8300.0)  # Bulk modulus, MPa

        self.RT = self.R_gas * self.theta0

        ##Create Function Spaces for problem
        self.U2 = ufl.VectorElement(
            "Lagrange", domain.ufl_cell(), 2
        )  # For displacement
        self.P1 = ufl.FiniteElement("Lagrange", domain.ufl_cell(), 1)  # For  pressure

        self.TH = ufl.MixedElement(
            [self.U2, self.P1, self.P1]
        )  # Taylor-Hood style mixed element

        self.ME = FunctionSpace(domain, self.TH)  # Total space for all DOFs

        self.w = Function(self.ME)
        self.u, self.mu, self.c = split(self.w)

        self.w_old = Function(self.ME)
        self.u_old, self.mu_old, self.c_old = split(self.w_old)

        self.w_old_2 = Function(self.ME)
        self.u_old_2, self.mu_old_2, self.c_old_2 = split(self.w_old_2)

        self.u_test, self.mu_test, self.c_test = TestFunctions(self.ME)

        self.dw = TrialFunction(self.ME)

        self.x = SpatialCoordinate(domain)
        self.domain = domain

        # Initialize problem
        if init_func is None:
            self._init_functions()
        else:
            init_func(self)

    def _init_functions(self):
        np.random.seed(2 + MPI.COMM_WORLD.Get_rank())

        # Some quick definitions to init random values
        V, _ = self.ME.sub(1).collapse()
        cBar_rand = Function(V)

        cBar_rand.interpolate(
            lambda x: 0.63 + 0.05 * (0.5 - np.random.rand(x.shape[1]))
        )  # Create a nice random value we can use

        fc_rand = self.RT * (
            ln(cBar_rand / (1 - cBar_rand)) + self.chi * (1 - 2 * cBar_rand)
        )  # use that relation to initate the two different sub expressions

        concentration = Expression(
            self.Omega * self.cMax * cBar_rand,
            self.ME.sub(2).element.interpolation_points(),
        )
        self.w.sub(2).interpolate(concentration)

        chemical_potential = Expression(
            fc_rand / self.RT, self.ME.sub(1).element.interpolation_points()
        )
        self.w.sub(1).interpolate(chemical_potential)

        self.w.sub(0).interpolate(lambda x: np.zeros((2, x.shape[1])))

        self.w_old.x.array[:] = self.w.x.array
        self.w_old_2.x.array[:] = self.w.x.array

    def WeakForms(self, dk: Constant):
        dx = ufl.dx(metadata={"quadrature_degree": 4})

        Res_u = inner(self.Tr, self.ax_grad_vector(self.u_test)) * self.x[0] * dx

        cdot = (3 * self.c - 4 * self.c_old + self.c_old_2) / (2 * dk)
        # The weak form for the mass balance of mobile species
        Res_0 = (
            dot(cdot, self.mu_test) * self.x[0] * dx
            - self.Omega
            * dot(self.Jmat, self.ax_grad_scalar(self.mu_test))
            * self.x[0]
            * dx
        )

        # The weak form for the concentration
        Res_1 = (
            dot(self.mu - self.fc / self.RT, self.c_test) * self.x[0] * dx
            - dot(
                (self.lam / self.RT) * self.ax_grad_scalar(self.cBar),
                self.ax_grad_scalar(self.c_test),
            )
            * self.x[0]
            * dx
        )
        # Total weak form
        self.Res = Res_0 + Res_1 + Res_u

        self.Res_mu = Res_0
        self.Res_c = Res_1
        self.Res_u = Res_u
        # Automatic differentiation tangent:
        self.a = derivative(self.Res, self.w, self.dw)

    def Kinematics(self):
        self.F = self.F_ax_calc(self.u)

        self.C = self.F.T * self.F  # Right Cauchy-Green tensor

        self.J = det(self.F)  # Total volumetric jacobian

        self.Js = 1.0 + self.c  # Solvent volumetric Jacobian

        self.Fs = self.Fs_ax_calc(self.Js)  # Solvent deformation gradient

        self.Fe = self.F * inv(self.Fs)  # Elastic deformation gradient

        self.Je = det(self.Fe)  # Elastic volumetric Jacobian

        self.Ce = self.Fe.T * self.Fe  # Elastic right Cauchy-Green tensor

        self.Ce_bar = self.Je ** (-2 / 3) * self.Ce  # Elastic isochoric tensor

        # #  Normalized Piola stress
        self.Te = self.Piola_calc(self.Je, self.Ce_bar, self.Ce)

        # Mandel Stress
        self.Me = self.Ce * self.Te

        # Calculate the Cauchy Stress
        self.T = inv(self.Je) * self.Fe * self.Te * inv(self.Fe)

        # Normalized Piola Stress

        self.Tr = (1 / self.Gshear) * self.J * self.T * inv(self.F).T

        self.cBar = self.c / (self.Omega * self.cMax)  # normalized concentration

        # Calculate the Species flux
        self.Jmat = self.Flux_calc(self.mu, self.c)
        # Calculate the f^c term
        self.fc = self.fc_calc(self.cBar, self.Me)

        # Kinematics

    # Gradient of vector field u
    def ax_grad_vector(self, u):
        grad_u = grad(u)
        dir3 = conditional(eq(self.x[0], 0), 0, u[0] / self.x[0])

        return as_tensor(
            [
                [grad_u[0, 0], grad_u[0, 1], 0],
                [grad_u[1, 0], grad_u[1, 1], 0],
                [0, 0, dir3],
            ]
        )

    # Gradient of scalar field y
    # (just need an extra zero for dimensions to work out)
    def ax_grad_scalar(self, y):
        grad_y = grad(y)
        return as_vector([grad_y[0], grad_y[1], 0.0])

    # ------------------------------------------------------------------------------
    # Species flux
    def Flux_calc(self, mu, c):
        #
        cBar = c / (self.Omega * self.cMax)  # normalized concentration
        #
        Mob = (self.D * c) / (self.Omega * self.RT) * (1 - cBar) * inv(self.C)
        #
        Jmat = -self.RT * Mob * self.ax_grad_scalar(mu)
        return Jmat

    # Calculate the f^c term
    def fc_calc(self, cBar, Me):
        #
        #
        fc = self.RT * (
            ln(cBar / (1 - cBar)) + self.chi * (1 - 2 * cBar)
        ) - self.Omega * (1 / 3 * tr(Me))
        #
        return fc

    # Eigenvalue decomposition of a 2D tensor
    def eigs(self, T):
        # Compute eigenvalues
        lambda1_0 = (
            T[0, 0] / 2
            + T[1, 1] / 2
            - sqrt(
                T[0, 0] ** 2
                - 2 * T[0, 0] * T[1, 1]
                + 4 * T[0, 1] * T[1, 0]
                + T[1, 1] ** 2
            )
            / 2
        )
        lambda2_0 = (
            T[0, 0] / 2
            + T[1, 1] / 2
            + sqrt(
                T[0, 0] ** 2
                - 2 * T[0, 0] * T[1, 1]
                + 4 * T[0, 1] * T[1, 0]
                + T[1, 1] ** 2
            )
            / 2
        )

        # Compute eigenvectors
        v11 = -T[1, 1] + lambda1_0
        v12 = T[1, 0]

        v21 = -T[1, 1] + lambda2_0
        v22 = T[1, 0]

        vec1_0 = as_vector([v11, v12])
        vec2_0 = as_vector([v21, v22])

        # Normalize eigenvectors
        vec1_0 = vec1_0 / sqrt(dot(vec1_0, vec1_0))
        vec2_0 = vec2_0 / sqrt(dot(vec2_0, vec2_0))

        # Reorder eigenvectors and eigenvalues
        vec1 = conditional(ge(lambda1_0, lambda2_0), vec1_0, vec2_0)
        vec2 = conditional(ge(lambda1_0, lambda2_0), vec2_0, vec1_0)

        lambda1 = conditional(ge(lambda1_0, lambda2_0), lambda1_0, lambda2_0)
        lambda2 = conditional(ge(lambda1_0, lambda2_0), lambda2_0, lambda1_0)

        return lambda1, lambda2, vec1, vec2

    def F_ax_calc(self, u):
        dim = len(u)
        Id = Identity(dim)  # Identity tensor
        val = grad(u)
        F = Id + val  # 2D Deformation gradient

        F33 = conditional(eq(self.x[0], 0), 1.0 + val[0, 0], 1.0 + (u[0]) / self.x[0])

        return as_tensor(
            [[F[0, 0], F[0, 1], 0.0], [F[1, 0], F[1, 1], 0.0], [0.0, 0.0, F33]]
        )  # Full pe F

    def Piola_calc(self, Je, Ce_bar, Ce):
        Tmat = Je ** (-2 / 3) * self.Gshear * (
            Identity(3) - (1 / 3) * tr(Ce_bar) * inv(Ce_bar)
        ) + self.K * Je * (Je - 1) * inv(Ce)

        return Tmat

    def Fs_ax_calc(self, Js):
        Fs = Js ** (1 / 3) * Identity(3)
        return Fs


class Cahn_Hillard_Plane_Strain:
    def __init__(self, domain, **kwargs):
        """
        Initializes a Cahn-Hilliard model with the given domain.

        Args:
            domain (dolfin.Mesh): The mesh domain for the model.
            **kwargs: Additional keyword arguments to be passed to the method.
        """

    def __init__(self, domain, **kwargs):
        # Create Contants

        self.Omega = Constant(domain, 4.05)  # Molar volume, um^3/pmol
        self.D = Constant(domain, 1e-2)  # Diffusivity, um^2/s
        self.chi = Constant(domain, 3.0)  # Phase parameter, (-)
        self.cMax = Constant(domain, 2.29e-2)  # Saturation concentration, pmol/um^3
        self.lam = Constant(domain, 5.5749e-1)  # Interface parameter, (pJ/pmol) um^2
        self.theta0 = Constant(domain, 298.0)  # Reference temperature, K
        self.R_gas = Constant(domain, 8.3145)  # Gas constant, pJ/(pmol K)

        self.Gshear = Constant(domain, 4980.0)  # Shear modulus, MPa
        self.K = Constant(domain, 8300.0)  # Bulk modulus, MPa

        self.RT = self.R_gas * self.theta0

        ##Create Function Spaces for problem
        self.U2 = ufl.VectorElement(
            "Lagrange", domain.ufl_cell(), 2
        )  # For displacement
        self.P1 = ufl.FiniteElement("Lagrange", domain.ufl_cell(), 1)  # For  pressure

        self.TH = ufl.MixedElement(
            [self.U2, self.P1, self.P1]
        )  # Taylor-Hood style mixed element

        self.ME = FunctionSpace(domain, self.TH)  # Total space for all DOFs

        self.w = Function(self.ME)
        self.u, self.mu, self.c = split(self.w)

        self.w_old = Function(self.ME)
        self.u_old, self.mu_old, self.c_old = split(self.w_old)

        self.u_test, self.mu_test, self.c_test = TestFunctions(self.ME)

        self.w_old_2 = Function(self.ME)
        self.u_old_2, self.mu_old_2, self.c_old_2 = split(self.w_old_2)

        self.dw = TrialFunction(self.ME)

        self.x = SpatialCoordinate(domain)
        self.domain = domain

        # Initialize problem
        self._init_functions()

    def _init_functions(self):
        np.random.seed(2 + MPI.COMM_WORLD.Get_rank())

        # Some quick definitions to init random values
        V, _ = self.ME.sub(1).collapse()
        cBar_rand = Function(V)

        cBar_rand.interpolate(
            lambda x: 0.63 + 0.05 * (0.5 - np.random.rand(x.shape[1]))
        )  # Create a nice random value we can use

        fc_rand = self.RT * (
            ln(cBar_rand / (1 - cBar_rand)) + self.chi * (1 - 2 * cBar_rand)
        )  # use that relation to initate the two different sub expressions

        concentration = Expression(
            self.Omega * self.cMax * cBar_rand,
            self.ME.sub(2).element.interpolation_points(),
        )
        self.w.sub(2).interpolate(concentration)

        chemical_potential = Expression(
            fc_rand / self.RT, self.ME.sub(1).element.interpolation_points()
        )
        self.w.sub(1).interpolate(chemical_potential)

        self.w.sub(0).interpolate(lambda x: np.zeros((2, x.shape[1])))

        self.w_old.x.array[:] = self.w.x.array
        self.w_old_2.x.array[:] = self.w.x.array

    def WeakForms(self, dk: Constant):
        dx = ufl.dx(metadata={"quadrature_degree": 4})

        Res_u = inner(self.Tr, self.pe_grad_vector(self.u_test)) * dx
        cdot = (3 * self.c - 4 * self.c_old + self.c_old_2) / (2 * dk)
        # The weak form for the mass balance of mobile species
        Res_0 = (
            dot(cdot, self.mu_test) * dx
            - self.Omega * dot(self.Jmat, self.pe_grad_scalar(self.mu_test)) * dx
        )

        # The weak form for the concentration
        Res_1 = (
            dot(self.mu - self.fc / self.RT, self.c_test) * dx
            - dot(
                (self.lam / self.RT) * self.pe_grad_scalar(self.cBar),
                self.pe_grad_scalar(self.c_test),
            )
            * dx
        )
        # Total weak form
        self.Res = Res_0 + Res_1 + Res_u

        # Automatic differentiation tangent:
        self.a = derivative(self.Res, self.w, self.dw)

    def Kinematics(self):
        self.F = self.F_pe_calc(self.u)

        self.C = self.F.T * self.F  # Right Cauchy-Green tensor

        self.J = det(self.F)  # Total volumetric jacobian

        self.Js = 1.0 + self.c  # Solvent volumetric Jacobian

        self.Fs = self.Fs_pe_calc(self.Js)  # Solvent deformation gradient

        self.Fe = self.F * inv(self.Fs)  # Elastic deformation gradient

        self.Je = det(self.Fe)  # Elastic volumetric Jacobian

        self.Ce = self.Fe.T * self.Fe  # Elastic right Cauchy-Green tensor

        self.Ce_bar = self.Je ** (-2 / 3) * self.Ce  # Elastic isochoric tensor

        # #  Normalized Piola stress
        self.Te = self.Piola_calc(self.Je, self.Ce_bar, self.Ce)

        # Mandel Stress
        self.Me = self.Ce * self.Te

        # Calculate the Cauchy Stress
        self.T = inv(self.Je) * self.Fe * self.Te * inv(self.Fe)

        # Normalized Piola Stress

        self.Tr = (1 / self.Gshear) * self.J * self.T * inv(self.F.T)

        self.cBar = self.c / (self.Omega * self.cMax)  # normalized concentration

        # Calculate the Species flux
        self.Jmat = self.Flux_calc(self.mu, self.c)
        # Calculate the f^c term
        self.fc = self.fc_calc(self.mu, self.c)

        # Kinematics

    # Gradient of vector field u
    def pe_grad_vector(self, u):
        grad_u = grad(u)
        return as_tensor(
            [
                [grad_u[0, 0], grad_u[0, 1], 0],
                [grad_u[1, 0], grad_u[1, 1], 0],
                [0, 0, 0],
            ]
        )

    # Gradient of scalar field y
    # (just need an extra zero for dimensions to work out)
    def pe_grad_scalar(self, y):
        grad_y = grad(y)
        return as_vector([grad_y[0], grad_y[1], 0.0])

    # ------------------------------------------------------------------------------
    # Species flux
    def Flux_calc(self, mu, c):
        #
        cBar = c / (self.Omega * self.cMax)  # normalized concentration
        #
        Mob = (self.D * c) / (self.Omega * self.RT) * (1 - cBar) * inv(self.C)
        #
        Jmat = -self.RT * Mob * self.pe_grad_scalar(mu)
        return Jmat

    # Calculate the f^c term
    def fc_calc(self, mu, c):
        #
        cBar = c / (self.Omega * self.cMax)  # normalized concentration
        #
        fc = self.RT * (
            ln(cBar / (1 - cBar)) + self.chi * (1 - 2 * cBar)
        ) - self.Omega * (1 / 3 * tr(self.Me))
        #
        return fc

    # Eigenvalue decomposition of a 2D tensor
    def eigs(self, T):
        # Compute eigenvalues
        lambda1_0 = (
            T[0, 0] / 2
            + T[1, 1] / 2
            - sqrt(
                T[0, 0] ** 2
                - 2 * T[0, 0] * T[1, 1]
                + 4 * T[0, 1] * T[1, 0]
                + T[1, 1] ** 2
            )
            / 2
        )
        lambda2_0 = (
            T[0, 0] / 2
            + T[1, 1] / 2
            + sqrt(
                T[0, 0] ** 2
                - 2 * T[0, 0] * T[1, 1]
                + 4 * T[0, 1] * T[1, 0]
                + T[1, 1] ** 2
            )
            / 2
        )

        # Compute eigenvectors
        v11 = -T[1, 1] + lambda1_0
        v12 = T[1, 0]

        v21 = -T[1, 1] + lambda2_0
        v22 = T[1, 0]

        vec1_0 = as_vector([v11, v12])
        vec2_0 = as_vector([v21, v22])

        # Normalize eigenvectors
        vec1_0 = vec1_0 / sqrt(dot(vec1_0, vec1_0))
        vec2_0 = vec2_0 / sqrt(dot(vec2_0, vec2_0))

        # Reorder eigenvectors and eigenvalues
        vec1 = conditional(ge(lambda1_0, lambda2_0), vec1_0, vec2_0)
        vec2 = conditional(ge(lambda1_0, lambda2_0), vec2_0, vec1_0)

        lambda1 = conditional(ge(lambda1_0, lambda2_0), lambda1_0, lambda2_0)
        lambda2 = conditional(ge(lambda1_0, lambda2_0), lambda2_0, lambda1_0)

        return lambda1, lambda2, vec1, vec2

    def F_pe_calc(self, u):
        dim = len(u)
        Id = Identity(dim)  # Identity tensor
        F = Id + grad(u)  # 2D Deformation gradient
        return as_tensor(
            [[F[0, 0], F[0, 1], 0], [F[1, 0], F[1, 1], 0], [0, 0, 1]]
        )  # Full pe F

    def Piola_calc(self, Je, Ce_bar, Ce):
        Tmat = Je ** (-2 / 3) * self.Gshear * (
            Identity(3) - (1 / 3) * tr(Ce_bar) * inv(Ce_bar)
        ) + self.K * Je * (Je - 1) * inv(Ce)

        return Tmat

    def Fs_pe_calc(self, Js):
        Fs = Js ** (1 / 3) * Identity(2)
        return as_tensor([[Fs[0, 0], Fs[0, 1], 0], [Fs[1, 0], Fs[1, 1], 0], [0, 0, 1]])


class Cahn_Hillard_Plane_Strain_no_mech:
    def __init__(self, domain, **kwargs):
        """
        Initializes a Cahn-Hilliard model with the given domain.

        Args:
            domain (dolfin.Mesh): The mesh domain for the model.
            **kwargs: Additional keyword arguments to be passed to the method.
        """

    def __init__(self, domain, **kwargs):
        # Create Contants

        self.Omega = Constant(domain, 4.05)  # Molar volume, um^3/pmol
        self.D = Constant(domain, 1e-2)  # Diffusivity, um^2/s
        self.chi = Constant(domain, 3.0)  # Phase parameter, (-)
        self.cMax = Constant(domain, 2.29e-2)  # Saturation concentration, pmol/um^3
        self.lam = Constant(domain, 5.5749e-1)  # Interface parameter, (pJ/pmol) um^2
        self.theta0 = Constant(domain, 298.0)  # Reference temperature, K
        self.R_gas = Constant(domain, 8.3145)  # Gas constant, pJ/(pmol K)

        self.RT = self.R_gas * self.theta0

        ##Create Function Spaces for problem
        self.U2 = ufl.VectorElement(
            "Lagrange", domain.ufl_cell(), 2
        )  # For displacement
        self.P1 = ufl.FiniteElement("Lagrange", domain.ufl_cell(), 1)  # For  pressure

        self.TH = ufl.MixedElement(
            [self.P1, self.P1]
        )  # Taylor-Hood style mixed element

        self.ME = FunctionSpace(domain, self.TH)  # Total space for all DOFs

        self.w = Function(self.ME)
        self.mu, self.c = split(self.w)

        self.w_old = Function(self.ME)
        self.mu_old, self.c_old = split(self.w_old)

        self.w_old_2 = Function(self.ME)
        self.mu_old_2, self.c_old_2 = split(self.w_old_2)

        self.mu_test, self.c_test = TestFunctions(self.ME)

        self.dw = TrialFunction(self.ME)

        self.x = SpatialCoordinate(domain)
        self.domain = domain

        # Initialize problem
        self._init_functions()

    def _init_functions(self):
        np.random.seed(2 + MPI.COMM_WORLD.Get_rank())

        # Some quick definitions to init random values
        V, _ = self.ME.sub(0).collapse()
        cBar_rand = Function(V)

        cBar_rand.interpolate(
            lambda x: 0.63 + 0.05 * (0.5 - np.random.rand(x.shape[1]))
        )  # Create a nice random value we can use

        fc_rand = self.RT * (
            ln(cBar_rand / (1 - cBar_rand)) + self.chi * (1 - 2 * cBar_rand)
        )  # use that relation to initate the two different sub expressions

        concentration = Expression(
            self.Omega * self.cMax * cBar_rand,
            self.ME.sub(1).element.interpolation_points(),
        )
        self.w.sub(1).interpolate(concentration)

        chemical_potential = Expression(
            fc_rand / self.RT, self.ME.sub(0).element.interpolation_points()
        )
        self.w.sub(0).interpolate(chemical_potential)

        self.w_old.x.array[:] = self.w.x.array
        self.w_old_2.x.array[:] = self.w.x.array

    def WeakForms(self, dk: Constant):
        dx = ufl.dx(metadata={"quadrature_degree": 4})

        cdot = (3 * self.c - 4 * self.c_old + self.c_old_2) / (2 * dk)
        # The weak form for the mass balance of mobile species
        Res_0 = (
            dot(cdot, self.mu_test) * dx
            - self.Omega * dot(self.Jmat, self.pe_grad_scalar(self.mu_test)) * dx
        )

        # The weak form for the concentration
        Res_1 = (
            dot(self.mu - self.fc / self.RT, self.c_test) * dx
            - dot(
                (self.lam / self.RT) * self.pe_grad_scalar(self.cBar),
                self.pe_grad_scalar(self.c_test),
            )
            * dx
        )
        # Total weak form
        self.Res = Res_0 + Res_1

        # Automatic differentiation tangent:
        self.a = derivative(self.Res, self.w, self.dw)

    def Kinematics(self):
        # Kinematics
        # self.F = self.F_pe_calc(self.u)
        # self.J = det(self.F)  # Total volumetric jacobian

        # # # Elastic volumetric Jacobian
        # self.Je = self.Je_calc(self.u, self.c)
        # # self.Je_old = self.Je_calc(self.u_old, self.c_old)

        # # #  Normalized Piola stress
        # self.Tmat = self.Piola_calc(self.u, self.p)

        # # #  Normalized species  flux
        # # self.Jmat = self.Flux_calc(self.u, self.mu, self.c)

        # Calculate the normalized concentration cBar
        self.cBar = self.c / (self.Omega * self.cMax)  # normalized concentration

        # Calculate the Species flux
        self.Jmat = self.Flux_calc(self.mu, self.c)

        # Calculate the f^c term
        self.fc = self.fc_calc(self.mu, self.c)

    # Gradient of vector field u
    def pe_grad_vector(self, u):
        grad_u = grad(u)
        return as_tensor(
            [
                [grad_u[0, 0], grad_u[0, 1], 0],
                [grad_u[1, 0], grad_u[1, 1], 0],
                [0, 0, 0],
            ]
        )

    # Gradient of scalar field y
    # (just need an extra zero for dimensions to work out)
    def pe_grad_scalar(self, y):
        grad_y = grad(y)
        return as_vector([grad_y[0], grad_y[1], 0.0])

    # ------------------------------------------------------------------------------
    # Species flux
    def Flux_calc(self, mu, c):
        #
        cBar = c / (self.Omega * self.cMax)  # normalized concentration
        #
        Mob = (self.D * c) / (self.Omega * self.RT) * (1 - cBar)
        #
        Jmat = -self.RT * Mob * self.pe_grad_scalar(mu)
        return Jmat

    # Calculate the f^c term
    def fc_calc(self, mu, c):
        #
        cBar = c / (self.Omega * self.cMax)  # normalized concentration
        #
        fc = self.RT * (ln(cBar / (1 - cBar)) + self.chi * (1 - 2 * cBar))
        #
        return fc

    # Eigenvalue decomposition of a 2D tensor
    def eigs(self, T):
        # Compute eigenvalues
        lambda1_0 = (
            T[0, 0] / 2
            + T[1, 1] / 2
            - sqrt(
                T[0, 0] ** 2
                - 2 * T[0, 0] * T[1, 1]
                + 4 * T[0, 1] * T[1, 0]
                + T[1, 1] ** 2
            )
            / 2
        )
        lambda2_0 = (
            T[0, 0] / 2
            + T[1, 1] / 2
            + sqrt(
                T[0, 0] ** 2
                - 2 * T[0, 0] * T[1, 1]
                + 4 * T[0, 1] * T[1, 0]
                + T[1, 1] ** 2
            )
            / 2
        )

        # Compute eigenvectors
        v11 = -T[1, 1] + lambda1_0
        v12 = T[1, 0]

        v21 = -T[1, 1] + lambda2_0
        v22 = T[1, 0]

        vec1_0 = as_vector([v11, v12])
        vec2_0 = as_vector([v21, v22])

        # Normalize eigenvectors
        vec1_0 = vec1_0 / sqrt(dot(vec1_0, vec1_0))
        vec2_0 = vec2_0 / sqrt(dot(vec2_0, vec2_0))

        # Reorder eigenvectors and eigenvalues
        vec1 = conditional(ge(lambda1_0, lambda2_0), vec1_0, vec2_0)
        vec2 = conditional(ge(lambda1_0, lambda2_0), vec2_0, vec1_0)

        lambda1 = conditional(ge(lambda1_0, lambda2_0), lambda1_0, lambda2_0)
        lambda2 = conditional(ge(lambda1_0, lambda2_0), lambda2_0, lambda1_0)

        return lambda1, lambda2, vec1, vec2


class Cahn_Hillard_3D_no_mech:
    def __init__(self, domain, **kwargs):
        """
        Initializes a Cahn-Hilliard model with the given domain.

        Args:
            domain (dolfin.Mesh): The mesh domain for the model.
            **kwargs: Additional keyword arguments to be passed to the method.
        """

    def __init__(self, domain, **kwargs):
        # Create Contants

        self.Omega = Constant(domain, 4.05)  # Molar volume, um^3/pmol
        self.D = Constant(domain, 1e-2)  # Diffusivity, um^2/s
        self.chi = Constant(domain, 3.0)  # Phase parameter, (-)
        self.cMax = Constant(domain, 2.29e-2)  # Saturation concentration, pmol/um^3
        self.lam = Constant(domain, 5.5749e-1)  # Interface parameter, (pJ/pmol) um^2
        self.theta0 = Constant(domain, 298.0)  # Reference temperature, K
        self.R_gas = Constant(domain, 8.3145)  # Gas constant, pJ/(pmol K)

        self.RT = self.R_gas * self.theta0

        ##Create Function Spaces for problem
        self.U2 = ufl.VectorElement(
            "Lagrange", domain.ufl_cell(), 2
        )  # For displacement
        self.P1 = ufl.FiniteElement("Lagrange", domain.ufl_cell(), 1)  # For  pressure

        self.TH = ufl.MixedElement(
            [self.P1, self.P1]
        )  # Taylor-Hood style mixed element

        self.ME = FunctionSpace(domain, self.TH)  # Total space for all DOFs

        self.w = Function(self.ME)
        self.mu, self.c = split(self.w)

        self.w_old = Function(self.ME)
        self.mu_old, self.c_old = split(self.w_old)

        self.w_old_2 = Function(self.ME)
        self.mu_old_2, self.c_old_2 = split(self.w_old_2)

        self.mu_test, self.c_test = TestFunctions(self.ME)

        self.dw = TrialFunction(self.ME)

        self.x = SpatialCoordinate(domain)
        self.domain = domain

        # Initialize problem
        self._init_functions()

    def _init_functions(self):
        np.random.seed(2 + MPI.COMM_WORLD.Get_rank())

        # Some quick definitions to init random values
        V, _ = self.ME.sub(0).collapse()
        cBar_rand = Function(V)

        cBar_rand.interpolate(
            lambda x: 0.63 + 0.05 * (0.5 - np.random.rand(x.shape[1]))
        )  # Create a nice random value we can use

        fc_rand = self.RT * (
            ln(cBar_rand / (1 - cBar_rand)) + self.chi * (1 - 2 * cBar_rand)
        )  # use that relation to initate the two different sub expressions

        concentration = Expression(
            self.Omega * self.cMax * cBar_rand,
            self.ME.sub(1).element.interpolation_points(),
        )
        self.w.sub(1).interpolate(concentration)

        chemical_potential = Expression(
            fc_rand / self.RT, self.ME.sub(0).element.interpolation_points()
        )
        self.w.sub(0).interpolate(chemical_potential)

        self.w_old.x.array[:] = self.w.x.array
        self.w_old_2.x.array[:] = self.w.x.array

    def WeakForms(self, dk: Constant):
        dx = ufl.dx(metadata={"quadrature_degree": 4})

        cdot = (3 * self.c - 4 * self.c_old + self.c_old_2) / (2 * dk)
        # The weak form for the mass balance of mobile species
        Res_0 = (
            dot(cdot, self.mu_test) * dx
            - self.Omega * dot(self.Jmat, grad(self.mu_test)) * dx
        )

        # The weak form for the concentration
        Res_1 = (
            dot(self.mu - self.fc / self.RT, self.c_test) * dx
            - dot(
                (self.lam / self.RT) * grad(self.cBar),
                grad(self.c_test),
            )
            * dx
        )
        # Total weak form
        self.Res = Res_0 + Res_1

        # Automatic differentiation tangent:
        self.a = derivative(self.Res, self.w, self.dw)

    def Kinematics(self):
        # Kinematics
        # self.F = self.F_pe_calc(self.u)
        # self.J = det(self.F)  # Total volumetric jacobian

        # # # Elastic volumetric Jacobian
        # self.Je = self.Je_calc(self.u, self.c)
        # # self.Je_old = self.Je_calc(self.u_old, self.c_old)

        # # #  Normalized Piola stress
        # self.Tmat = self.Piola_calc(self.u, self.p)

        # # #  Normalized species  flux
        # # self.Jmat = self.Flux_calc(self.u, self.mu, self.c)

        # Calculate the normalized concentration cBar
        self.cBar = self.c / (self.Omega * self.cMax)  # normalized concentration

        # Calculate the Species flux
        self.Jmat = self.Flux_calc(self.mu, self.c)

        # Calculate the f^c term
        self.fc = self.fc_calc(self.mu, self.c)

    # Gradient of vector field u
    # ------------------------------------------------------------------------------
    # Species flux
    def Flux_calc(self, mu, c):
        #
        cBar = c / (self.Omega * self.cMax)  # normalized concentration
        #
        Mob = (self.D * c) / (self.Omega * self.RT) * (1 - cBar)
        #
        Jmat = -self.RT * Mob * grad(mu)
        return Jmat

    # Calculate the f^c term
    def fc_calc(self, mu, c):
        #
        cBar = c / (self.Omega * self.cMax)  # normalized concentration
        #
        fc = self.RT * (ln(cBar / (1 - cBar)) + self.chi * (1 - 2 * cBar))
        #
        return fc

    # Eigenvalue decomposition of a 2D tensor
    def eigs(self, T):
        # Compute eigenvalues
        lambda1_0 = (
            T[0, 0] / 2
            + T[1, 1] / 2
            - sqrt(
                T[0, 0] ** 2
                - 2 * T[0, 0] * T[1, 1]
                + 4 * T[0, 1] * T[1, 0]
                + T[1, 1] ** 2
            )
            / 2
        )
        lambda2_0 = (
            T[0, 0] / 2
            + T[1, 1] / 2
            + sqrt(
                T[0, 0] ** 2
                - 2 * T[0, 0] * T[1, 1]
                + 4 * T[0, 1] * T[1, 0]
                + T[1, 1] ** 2
            )
            / 2
        )

        # Compute eigenvectors
        v11 = -T[1, 1] + lambda1_0
        v12 = T[1, 0]

        v21 = -T[1, 1] + lambda2_0
        v22 = T[1, 0]

        vec1_0 = as_vector([v11, v12])
        vec2_0 = as_vector([v21, v22])

        # Normalize eigenvectors
        vec1_0 = vec1_0 / sqrt(dot(vec1_0, vec1_0))
        vec2_0 = vec2_0 / sqrt(dot(vec2_0, vec2_0))

        # Reorder eigenvectors and eigenvalues
        vec1 = conditional(ge(lambda1_0, lambda2_0), vec1_0, vec2_0)
        vec2 = conditional(ge(lambda1_0, lambda2_0), vec2_0, vec1_0)

        lambda1 = conditional(ge(lambda1_0, lambda2_0), lambda1_0, lambda2_0)
        lambda2 = conditional(ge(lambda1_0, lambda2_0), lambda2_0, lambda1_0)

        return lambda1, lambda2, vec1, vec2

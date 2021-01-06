# -*- coding: utf-8 -*-
"""
===============================================================================
   OPTIMAL CONTROL OF THE NAVIER-STOKES EQUATION WITH DELAY IN CONVECTION
===============================================================================

This Python script approximates the solution of the following optimal control
problem for the Navier-Stokes equation with delay in the convection term.
The problem is given by:

    min (1/2){alpha_O |u - ud|^2 + alpha_T |u(T) - uT|^2
              + alpha_X |Curl u|^2 + alpha |q|^2}
    subject to
        q in L^2((0, T) times Omega)
        u_t - Delta u + (u(.-r).Grad)u + Grad p = f + q, in (0, T) times Omega
                                          div u = 0,     in (0, T) times Omega
                                              u = 0,     on (0, T) times Gamma
    with the initial data
        u(0) = u0,      in Omega
    and initial history
        u(s) = z0(s),   in (-r, 0) times Omega.

Here, u, p, ud, uT and q are the fluid velocity, fluid pressure, desired
velocity, desired velocity at the terminal time and control, respectively.
The default domain Omega is the unit square (0, 1)^2. Spatial discretization
is based on the Taylor-Hood (P2/P1) or P1Bubble/P1 triangular elements, while
for the time discretization, an Implicit-Explicit (IMEX) Euler scheme is
utilized. In the case where there is no delay (r = 0), the scheme reduces to
the usual Navier-Stoke equation.

The three standard python packages NumPy, SciPy and Matplotlib are required in
the implementation. Matrices for the finite element assembly are obtained using
the accompanied module stokesfem.py.

For more details, refer to the manuscript:
    'Optimal Control for the Navier-Stokes Equation with Time Delay in the
    Convection: Analysis and Finite Element Approximations' by Gilbert Peralta
    and John Sebastian Simon, to appear in Journal of Mathematical Fluid
    Mechanics.


Gilbert Peralta
Department of Mathematics and Computer Science
University of the Philippines Baguio
Governor Pack Road, Baguio, Philippines 2600
Email: grperalta@up.edu.ph
7 January 2021
"""

from __future__ import division
from time import time
from scipy import sparse as sp
from scipy.sparse.linalg import splu
import numpy as np
import stokesfem as sfem
import utils

__author__ = "Gilbert Peralta"
__copyright__ = "Copyright 2020, Gilbert Peralta"
__version__ = "1.0"
__institution__ = "University of the Philippines Baguio"
__email__ = "grperalta@up.edu.ph"
__date__ = "7 January 2021"

#-----------------------------------------------------------------------------------
# PARAMETERS, TERMPORAL GRID, FUNCTIONS AND DATA CLASSES
#-----------------------------------------------------------------------------------

class Parameters():
    """
    Class of parameters for the optimal control problem (ocp).

    Attributes
    ----------
    T : float
        Time horizon.
    nu : float
        Fluid viscosity.
    Nt : int
        Number of time steps.
    tau : float
        Delay parameter.
    Pen : float
        Artificial compressibility penalty parameter.
    alpha : float
        Tikhonov regularization parameter.
    alpha_O : float
        Coefficient of velocity tracking.
    alpha_T : float
        Coefficient of terminal velocity tracking.
    alpha_X : float
        Coefficient of vorticity tracking.
    ocptol : float
        Tolerance for the Barzilai-Borwein (BB) gradient algorithm.
    ocpmaxit : int
        Maximum number of iterations for the BB gradient method.
    """
    def __init__(self, T, Nt, nu, dt, tau, alpha, alpha_O, alpha_T,
        alpha_X, ocptol, ocpmaxit, pen):
        """
        Class initialization.
        """
        self.T = T
        self.Nt = Nt
        self.nu = nu
        self.dt = dt
        self.tau = tau
        self.pen = pen
        self.alpha = alpha
        self.alpha_O = alpha_O
        self.alpha_T = alpha_T
        self.alpha_X = alpha_X
        self.ocptol = ocptol
        self.ocpmaxit = ocpmaxit

    def __str__(self):
        """
        String representation.
        """
        LINE = utils.Colors.UNDERLINE + ' '*80 + utils.Colors.ENDC
        print(utils.Colors.BOLD + utils.Colors.BLUE
            + '\t\t\t\tPARAMETERS' + utils.Colors.ENDC)
        string = LINE + '\nAttribute\t Description \t\t\t\t Value\n' + LINE
        string += (
            '\nT\t\t Time Horizon \t\t\t\t {:.2f}\n'
            + 'nu\t\t Viscosity \t\t\t\t {:.2f}\n'
            + 'Nt\t\t Number of time steps \t\t\t {}\n'
            + 'dt\t\t Temporal grid size \t\t\t {}\n'
            + 'tau\t\t Delay \t\t\t\t\t {:.2f}\n'
            + 'pen\t\t Artificial compressibility \t\t {:.2e}\n'
            + 'alpha\t\t Tikhonov regularization \t\t {:.2e}\n'
            + 'alpha_O\t\t Velocity tracking coefficient \t\t {:.2e}\n'
            + 'alpha_T\t\t Final tracking coefficient \t\t {:.2e}\n'
            + 'alpha_X\t\t Vorticity tracking coefficient\t\t {:.2e}\n'
            + 'ocptol\t\t OCP tolerance \t\t\t\t {:.2e}\n'
            + 'ocpmaxit\t OCP maximum number of iterations \t {}\n')
        string += LINE
        return string.format(self.T, self.nu, self.Nt, self.dt,
            self.tau, self.pen, self.alpha, self.alpha_O, self.alpha_T,
            self.alpha_X, self.ocptol, self.ocpmaxit, self.pen)

    @classmethod
    def get_default(self):
        """
        Default parameters for the optimal control problem.
        """
        self = Parameters(T=1.0, Nt=21, nu=1.0, dt=0.0, tau=0.5,
            alpha=0.1, alpha_O=1.0, alpha_T=1.0, alpha_X=1.0,
            ocptol=1e-6, ocpmaxit=1000, pen=1e-10)
        self.dt = self.T / (self.Nt - 1)
        return self


class TemporalGrid():
    """
    Class for temporal and history grids.

    Attributes
    ----------
    mesh : numpy.ndarray
        The temporal mesh.
    num_hist : int
        Number of subdivisions of the history interval.
    """
    def __init__(self, param):
        """
        Class initialization.

        Parameter
        ---------
        param : nsedelay.Parameters class
            Set of parameters for the optimal control problem.
        """
        self.mesh = np.linspace(0.0, param.T, param.Nt)
        self.num_hist = int(param.tau / param.dt)


class Functions():
    """
    Functions utilized in the generation of initial velocity, desired velocity
    and history.
    """
    def __init__(self):
        """
        Class initialization.
        """

    def u(self, x, y):
        """
        First component of initial velocity.
        """
        return (1.0 - np.cos(2*np.math.pi*x)) * np.sin(2*np.math.pi*y)

    def v(self, x, y):
        """
        Second component of initial velocity.
        """
        return np.sin(2*np.math.pi*x) * (np.cos(2*np.math.pi*y) - 1.0)

    def time_coeff(self, t):
        """
        Temporal coefficient for desired velocity.
        """
        return np.cos(np.math.pi*t)

    def hist_coeff(self, t, tau):
        """
        Temporal coefficient for initial history.
        """
        return self.time_coeff(t - tau)


class Data():
    """
    Class for initial velocity, history and target velocity.

    Attributes
    ----------
    init_u : numpy.ndarray
        Components of the initial velocity.
    hist_u : numpy.ndarray
        Components of the initial history.
    goal_u : numpy.ndarray
        Components of the desired velocity.
    """
    def __init__(self, init_u, hist_u, goal_u):
        """
        Class initialization.
        """
        self.init_u = init_u.astype(float)
        self.hist_u = hist_u.astype(float)
        self.goal_u = goal_u.astype(float)

    @classmethod
    def get_default(self, param, tgrid, mesh, femtype):
        """
        Default initial, history and desired data for the optimal
        control problem.

        Parameters
        ----------
        param : nsedelay.Parameters class
            Parameters of the optimal control problem.
        tgrid : nsedelay.TemporalGrid class
            Class for temporal and history grids.
        mesh : stokesfem.Mesh class
            Triangulation of the domain.
        femtype : str
            Either 'p1_bubble' or 'taylor_hood'.
        """
        fun = Functions()

        histgrid = 0.5 * (
            fun.hist_coeff(tgrid.mesh[:tgrid.num_hist], param.tau)
            + fun.hist_coeff(tgrid.mesh[1:tgrid.num_hist+1], param.tau))
        timegrid = 0.5 * (
            fun.time_coeff(tgrid.mesh[:-1]) + fun.time_coeff(tgrid.mesh[1:]))

        if femtype is 'taylor_hood':
            mesh_all_node = mesh.all_node()
            x, y = mesh_all_node[:, 0], mesh_all_node[:, 1]
            self.init_u = np.append(fun.u(x, y), fun.v(x, y))
        elif femtype is 'p1_bubble':
            self.init_u = np.append(sfem.bubble_interpolation(mesh, fun.u),
                sfem.bubble_interpolation(mesh, fun.v))

        self.goal_u = np.outer(self.init_u, timegrid)
        self.hist_u = np.outer(self.init_u, histgrid)

        return self

#-----------------------------------------------------------------------------------
# FINITE ELEMENT MATRICES
#-----------------------------------------------------------------------------------

def get_fem_matrices(mesh, param, femstruct, noslipbc):
    """
    Finite element matrices.

    Parameters
    ----------
    mesh : stokesfem.Mesh class
        Triangulation of the domain.
    param : nsedelay.Parameters class
        Parameters of the optimal control problem.
    femstruct : stokesfem.FEMDataStruct class
        Finite element data structure.
    noslipbc : numpy.ndarray
        List of indices corresponding to the no-slip boundary conditions.

    Returns
    -------
    Asolve : instance or method
        Solver for the system matrix.
    M : scipy.sparse_csr matrix
        Mass matrix.
    K : scipy.sparse_csr matrix
        Stiffness matrix.
    V : scipy.sparse_csr matrix
        Matrix associated with vorticity.
    TIME_MATRIX_ASSEMBLY : float
        Elapsed time for matrix assembly in seconds.
    TIME_LU_FACTORIZATION : float
        Elapsed time for sparse LU factorization in seconds.
    MATRIX_NNZ : int
        Number of nonzero entries in the system matrix.
    MATRIX_SHAPE : tuple
        Shape of system matrix.
    """
    # matrix assembly
    start = time()
    K, M, Bx, By, _ = sfem.assemble(mesh, femstruct)
    end = time()
    TIME_MATRIX_ASSEMBLY = end - start

    # system matrix
    A = (1.0/param.dt) * M + param.nu * K
    A = sp.kron(sp.identity(2), A).tocsc()
    B = sp.bmat([[Bx, By]], format='csc')
    A = A + (1.0/param.pen) * (B.T * B)
    A = sfem.apply_noslip_bc(A, noslipbc)

    MATRIX_NNZ = A.nnz
    MATRIX_SHAPE = A.shape

    # assembly of voriticity matrix
    V = sfem.vorticity(mesh, femstruct)

    # sparse LU factorization
    splu_opts = dict(DiagPivotThresh=1e-6, SymmetricMode=True,
        PivotGrowth=True)
    start = time()
    Asolve = sp.linalg.splu(A, permc_spec="MMD_AT_PLUS_A",
        panel_size=3, options=splu_opts).solve
    end = time()
    TIME_LU_FACTORIZATION = end - start

    return (Asolve, M, K, V, TIME_MATRIX_ASSEMBLY, TIME_LU_FACTORIZATION,
        MATRIX_NNZ, MATRIX_SHAPE)

#-----------------------------------------------------------------------------------
# OPTIMAL CONTROL PROBLEM (OCP) CLASS
#-----------------------------------------------------------------------------------

def Residual(u, goal_u):
    """
    Computes the difference of the state and desired state.

    Parameters
    ----------
    u : numpy.ndarray
        State variable.
    goal_u : numpy.ndarray
        Desired state variable.

    Returns
    -------
    numpy.ndarray
    """
    return u[:, 1:] - goal_u


def State_Solver(init_u, hist_u, control, param, mesh, tgrid, femstruct,
    noslipbc, Asolve, M, info=True):
    """
    Solves the state equation.

    Parameters
    ----------
    init_u : numpy.ndarray
        Initial velocity.
    hist_u : numpy.ndarray
        Initial history.
    control : numpy.ndarray
        Control variable.
    param : nsedelay.Parameters class
        Parameters of the optimal control problem.
    mesh : stokesfem.Mesh class
        Triangulation of the domain.
    tgrid : nsedelay.TemporalGrid class
        Temporal discretization.
    femstruct : stokesfem.FEMDataStruct class
        Finite element data structure.
    noslipbc : numpy.ndarray
        List of indices corresponding to the no-slip boundary condition.
    Asolve : instance
        Solver for the system matrix.
    M : scipy.sparse_csr matrix
        Mass matrix.
    info : boolean
        If True, then prints elapsed time of computing the solution.

    Returns
    -------
    numpy.ndarray
    """
    start = time()

    # pre-allocation
    u = np.zeros((2*mesh.dof, param.Nt)).astype(float)

    # input initial data
    u[:, 0] = init_u

    # time-advancing via IMEX Euler method
    for i in range(param.Nt-1):
        if i < tgrid.num_hist:
            # assembly of convection matrix using history
            N = sfem.convection(mesh, hist_u[:mesh.dof, i],
                hist_u[mesh.dof:, i], femstruct)
        else:
            # assembly of convection matrix using computed velocity
            N = sfem.convection(mesh, u[:mesh.dof, i - tgrid.num_hist],
                u[mesh.dof:, i - tgrid.num_hist], femstruct)
        # right hand side vector
        rhs_u = M * ((1/param.dt) * u[:mesh.dof, i]
            + control[:mesh.dof, i]) + N * u[:mesh.dof, i]
        rhs_v = M * ((1/param.dt) * u[mesh.dof:, i]
            + control[mesh.dof:, i]) + N * u[mesh.dof:, i]
        rhs = np.append(rhs_u, rhs_v)
        rhs[noslipbc] = 0

        # solve velocity at current time step
        u[:, i+1] = Asolve(rhs)

    end = time()
    if info:
        print("\tState solver elapsed time:\t{:.8e} seconds".format(end-start))

    return u


def Adjoint_Solver(res_u, hist_u, u, param, mesh, tgrid, femstruct, noslipbc,
    Asolve, M, V, info=True):
    """
    Solves the adjoint/dual equation.

    Parameters
    ----------
    res_u : numpy.ndarray
        Residual variable.
    hist_u : numpy.ndarray
        Initial history.
    u : numpy.ndarray
        State variable.
    param : nsedelay.Parameters class
        Parameters of the optimal control problem.
    mesh : stokesfem.Mesh class
        Triangulation of the domain.
    tgrid : TemporalGrid class
        Temporal discretization.
    femstruct : stokesfem.FEMDataStruct class
        Finite element data structure.
    noslipbc : numpy.ndarray
        List of indices corresponding to the no-slip boundary conditions.
    Asolve : instance
        Solver for the system matrix.
    M : scipy.sparse_csr matrix
        Mass matrix.
    V : scipy.sparse_csr matrix
        Matrix associated with vorticity.
    info : boolean
        If True, then prints elapsed time of computing the solution.

    Returns
    -------
    numpy.ndarray
    """
    start = time()

    # pre-allocation
    w = np.zeros((2*mesh.dof, param.Nt)).astype(float)

    # terminal data
    w[:, -1] = param.alpha_T * res_u[:, -1]

    # time-advancing scheme via IMEX Euler method
    for i in range(param.Nt - 2, -1, -1):
        if i < tgrid.num_hist:
            # assembly of convection matrix using history
            N = sfem.convection(mesh, hist_u[:mesh.dof, i],
                hist_u[mesh.dof:, i], femstruct)
        else:
            # assembly of convection matrix using computed velocity
            N = sfem.convection(mesh, u[:mesh.dof, i - tgrid.num_hist],
                u[mesh.dof:, i - tgrid.num_hist], femstruct)

        # temporary components of right hand vector
        rhs_w = M * (param.alpha_O * res_u[:mesh.dof, i]
            + (1/param.dt) * w[:mesh.dof, i+1]) + N.T * w[:mesh.dof, i+1]
        rhs_y = M * (param.alpha_O * res_u[mesh.dof:, i]
            + (1/param.dt) * w[mesh.dof:, i+1]) + N.T * w[mesh.dof:, i+1]

        if i >= param.Nt - tgrid.num_hist - 1:
            rhs = np.append(rhs_w, rhs_y)
        else:
            # assembly of dual convection matrix
            P = sfem.convection_dual(mesh, w[:mesh.dof, i+tgrid.num_hist+1],
                w[mesh.dof:, i+tgrid.num_hist+1], femstruct)
            rhs = np.append(rhs_w, rhs_y) + P * u[:, i+tgrid.num_hist+1]

        # include vorticity to the right hand vector
        if param.alpha_X > 0:
            rhs = rhs + param.alpha_X * V * u[:, i+1]
        rhs[noslipbc] = 0

        # solve dual velocity at current time step
        w[:, i] = Asolve(rhs)

    end = time()
    if info:
        print("\tAdjoint solver elapsed time:\t{:.8e} seconds".format(end-start))

    return w


def Cost_Functional(res_u, u, control, param, mesh, M, V):
    """
    Calculate the objective cost value.

    Parameters
    ----------
    res_u : numpy.ndarray
        Residual variable.
    u : numpy.ndarray
        State variable.
    control : numpy.ndarray
        Control variable.
    param : nsedelay.Parameters class
        Parameters of the optimal control problem.
    mesh : stokesfem.Mesh class
        Triangulation of the domain.
    M : scipy.sparse matrix
        Mass matrix.
    V : scipy.sparse matrix
        Matrix associated with vorticity.

    Returns
    -------
    float
    """
    # velocity tracking part
    J_O = np.sum(res_u[:mesh.dof, :] * (M * res_u[:mesh.dof, :])) \
        + np.sum(res_u[mesh.dof:, :] * (M * res_u[mesh.dof:, :]))

    # terminal velocity tracking part
    if param.alpha_T > 0:
        J_T = np.dot(res_u[:mesh.dof, -1], M * res_u[:mesh.dof, -1]) \
            + np.dot(res_u[mesh.dof:, -1], M * res_u[mesh.dof:, -1])
    else:
        J_T = 0.0

    # vorticity tracking part
    if param.alpha_X > 0:
        J_X = np.sum(u * (V * u))
    else:
        J_X = 0.0

    # Tikhonov regularization
    J_q = np.sum(control[:mesh.dof, :] * (M * control[:mesh.dof, :])) \
        + np.sum(control[mesh.dof:, :] * (M * control[mesh.dof:, :]))

    return 0.5 * param.dt * (param.alpha_O * J_O + param.alpha_X * J_X
        +  param.alpha * J_q) + 0.5 * param.alpha_T * J_T


def Adjoint_to_Control(w, null_control):
    """
    Maps the adjoint to control.

    Parameters
    ----------
    w : numpy.ndarray
        Adjoint variable.
    null_control : list
        List of indices where no control is applied.

    Returns
    -------
    numpy.ndarray
    """
    adj_to_control = w[:, :-1]
    adj_to_control[null_control, :] = 0.0

    return adj_to_control


def Cost_Derivative(control, adj_to_control, param):
    """
    Derivative of the cost functional.

    Parameters
    ----------
    control : numpy.ndarray
        Control variable.
    adjoint_to_control : numpy.ndarray
        Value of adjoint to control map.
    param : nsedelay.Parameters class
        Parameters of the optimal control problem.

    Returns
    -------
    numpy.ndarray
    """
    return param.alpha * control + adj_to_control


def L2_norm(var, param, mesh, M):
    """
    Returns the space-time L2-norm of var.

    Parameters
    ----------
    var : numpy.ndarray
        Array for the values of the variable at the space-time nodes.
    param : nsedelay.Parameters class
        Parameters of the optimal control problem.
    mesh : stokesfem.Mesh class
        Triangulation of the domain.
    M : scipy.sparse matrix
        Mass matrix.

    Returns
    -------
    float
    """
    norm = np.sum(var[:mesh.dof, :] * (M * var[:mesh.dof, :])) \
        + np.sum(var[mesh.dof:, :] * (M * var[mesh.dof:, :]))
    norm = np.sqrt(param.dt * norm)

    return norm


def L2H1_norm(var, param, mesh, K):
    """
    Returns the space-time L2(H1)-norm of var.

    Parameters
    ----------
    var : numpy.ndarray
        Array for the values of the variable at the space-time nodes.
    param : nsedelay.Parameters class
        Parameters of the optimal control problem.
    mesh : stokesfem.Mesh class
        Triangulation of the domain.
    K : scipy.sparse matrix
        Stiffness matrix.

    Returns
    -------
    float
    """
    norm = np.sum(var[:mesh.dof, :] * (K * var[:mesh.dof, :])) \
        + np.sum(var[mesh.dof:, :] * (K * var[mesh.dof:, :]))
    norm = np.sqrt(param.dt * norm)

    return norm


def LinfL2_norm(var, param, mesh, M):
    """
    Returns the space-time Linf(L2)-norm of var.

    Parameters
    ----------
    var : numpy.ndarray
        Array for the values of the variable at the space-time nodes.
    param : nsedelay.Parameters class
        Parameters of the optimal control problem.
    mesh : stokesfem.Mesh class
        Triangulation of the domain.
    M : scipy.sparse matrix
        Mass matrix.

    Returns
    -------
    float
    """
    L2_norm = np.zeros(var.shape[1]).astype(float)
    for i in range(len(L2_norm)):
        L2_norm[i] = np.sqrt(np.sum(var[:mesh.dof, i] * (M * var[:mesh.dof, i])
        + var[mesh.dof:, i] * (M * var[mesh.dof:, i])))
    return np.max(L2_norm)


class OCP():
    """
    Class for the optimal control problem (ocp).

    Attributes
    ----------
    femtype : str
        Either 'p1_bubble' or 'taylor_hood'.
    param : nsedelay.Parameters class
        Parameters of the ocp class.
    tgrid : nsedelay.TemporalGrid class
        Temporal and history grids.
    mesh : stokesfem.Mesh class
        Data structure for the triangulation of the domain.
    data : nsedelay.Data class
        Inital data, history and desired data.
    femstruct : stokesfem.FEMDataStruct class
        Finite element data structure.
    noslipbc : numpy.ndarray
        List of indices corresponding to the no-slip boundary conditions.
    null_control : list
        List of node indices where no control is applied.
    Asolve : instance
        Solver for the system matrix.
    M : scipy.sparse matrix
        Mass matrix.
    K : scipy.sparse.csr_matrix
        Stiffness matrix.
    V : scipy.sparse matrix
        Matrix associated with vorticity.
    init_control : numpy.ndarray
        Initial control (default set to zero).
    force : numpy.ndarray
        External source.
    TIME_MATRIX_ASSEMBLY : float
        Elapsed time for matrix assembly in seconds.
    TIME_LU_FACTORIZATION : float
        Elapsed time for sparse LU factorization in seconds.
    MATRIX_NNZ : int
        Number of nonzero entries in the system matrix.
    MATRIX_SHAPE : tuple
        Shape of system matrix.
    """
    def __init__(self, param=None, mesh=None, data=None, force=None,
        null_control=[], femtype='taylor_hood'):
        """
        Class initialization.
        """
        # finite element type
        self.femtype = femtype

        # null control indices
        self.null_control = null_control

        # set parameters for the optimal control problem
        if param is None:
            self.param = Parameters.get_default()
        else:
            self.param = param
            self.param.dt = self.param.T / (self.param.Nt-1)

        # temporal and history grids
        self.tgrid = TemporalGrid(self.param)

        # generate or assign mesh data structure
        if mesh is None:
            self.mesh = sfem.square_uniform_trimesh(11).femprocess(
                self.femtype)
        else:
            self.mesh = mesh.femprocess(self.femtype)

        # generate or assign initial and desired data
        if data is None:
            self.data = Data.get_default(self.param, self.tgrid, self.mesh,
                self.femtype)
        else:
            self.data = data

        # generate finite element data structure
        self.femstruct \
            = sfem.get_fem_data_struct(self.mesh, name=self.femtype)

        # determine Dirichlet nodes
        if self.femtype is 'taylor_hood':
            self.noslipbc = np.append(self.mesh.all_bdy_node(),
                self.mesh.dof + self.mesh.all_bdy_node())
        elif self.femtype is 'p1_bubble':
            self.noslipbc = np.append(self.mesh.bdy_node,
                self.mesh.dof + self.mesh.bdy_node)

        # assembly of total, mass, stiffness and vorticity matrices
        (self.Asolve, self.M, self.K, self.V, self.TIME_MATRIX_ASSEMBLY,
            self.TIME_LU_FACTORIZATION, self.MATRIX_NNZ, self.MATRIX_SHAPE) \
            = get_fem_matrices(self.mesh, self.param, self.femstruct, self.noslipbc)

        # initialize control
        self.init_control \
            = np.zeros((2*self.mesh.dof, self.param.Nt-1)).astype(float)

        # source fucntion
        self.force = force

    def state_solver(self, control):
        """
        Solves the state equation. See also <State_Solver>.
        """
        if self.force is not None:
            control = control + self.force

        return State_Solver(self.data.init_u, self.data.hist_u,
            control, self.param, self.mesh, self.tgrid, self.femstruct,
            self.noslipbc, self.Asolve, self.M)

    def residual(self, u):
        """
        Computes the residual. See also <Residual>.
        """
        return Residual(u, self.data.goal_u)

    def adjoint_solver(self, u, res_u):
        """
        Solves the adjoint equation. See also <Adjoint_Solver>.
        """
        return Adjoint_Solver(res_u, self.data.hist_u, u, self.param,
            self.mesh, self.tgrid, self.femstruct, self.noslipbc,
            self.Asolve, self.M, self.V)

    def der_cost(self, control, adj_to_control):
        """
        Returns the derivative of the cost functional. See also <Cost_Derivative>.
        """
        return Cost_Derivative(control, adj_to_control, self.param)

    def der_cost_norm(self, der):
        """
        Computes the L2-norm of the derivative of the cost functional.
        See also <L2_norm>.
        """
        return L2_norm(der, self.param, self.mesh, self.M)

    def cost(self, res_u, u, control):
        """
        Calculates the cost functional. See also <Cost_Functional>.
        """
        return Cost_Functional(res_u, u, control, self.param, self.mesh,
            self.M, self.V)

    def adjoint_to_control(self, w):
        """
        Maps the adjoint equation to control. See also <Adjoint_to_Control>.
        """
        return Adjoint_to_Control(w, self.null_control)

    def init_steplength(self, u, control):
        """
        Calculates the denominator of the steepest descent steplength.
        """
        return Cost_Functional(u, u, control, self.param, self.mesh,
            self.M, self.V)

    def print(self):
        """
        Displays certain data in the class.
        """
        # parameters
        print(self.param)

        # finite element type
        print(utils.Colors.BOLD + utils.Colors.BLUE
            + '\nFINITE ELEMENT METHOD' + utils.Colors.ENDC)
        print("\t{}".format(self.femtype.upper()))

        # mesh properties
        print(utils.Colors.BOLD + utils.Colors.BLUE
            + '\nMESH DATA STRUCTURE' + utils.Colors.ENDC)
        print('\tNumber of Nodes: {}'.format(self.mesh.num_node))
        print('\tNumber of Cells: {}'.format(self.mesh.num_cell))
        print('\tNumber of Edges: {}'.format(self.mesh.num_edge))
        print('\tNumber of Boundary Nodes: {}'.format(
            len(self.mesh.bdy_node)))
        print('\tMesh Size: {:.6f}'.format(self.mesh.size()))

        # matrix properties
        print(utils.Colors.BOLD + utils.Colors.BLUE
            + '\nSYSTEM MATRIX PROPERTIES' + utils.Colors.ENDC)
        print('\tShape: {}'.format(self.MATRIX_SHAPE))
        print('\tDensity: {:.6f}'.format(self.MATRIX_NNZ \
            / (self.MATRIX_SHAPE[0] * self.MATRIX_SHAPE[1])))
        print('\tNumber of Nonzero Entries: {}'.format(self.MATRIX_NNZ))
        print('\tElapsed time for matrix assembly:  {:.6e} seconds'.format(
            self.TIME_MATRIX_ASSEMBLY))
        print('\tElapsed time for LU factorization: {:.6e} seconds'.format(
            self.TIME_LU_FACTORIZATION))

#-----------------------------------------------------------------------------------
# BARZILAI-BORWEIN GRADIENT ALGORITHM
#-----------------------------------------------------------------------------------

def barzilai_borwein(ocp, version=1):
    """
    Barzilai-Borwein version of the gradient method.

    The algorithm stops if the consecutive cost function values have relative error
    less than the pescribed tolerance or the maximum number of iterations is reached.
    The second point of the gradient method is given by x = x - g, where x is the
    initial point and g is its gradient value.

    Parameters
    ----------
    ocp : nsedelay.OCP class
        Class for the optimal control problem.

    Returns
    -------
    state, control, adjoint, residue, COST_VALUES, OPT_REST_NORM, REL_ERROR
        : tuple of numpy.ndarray and lists
        numerical optimal state, control, adjoint state, residual,
        cost values, norm of optimality residual and relative error
    """

    string = (utils.Colors.BOLD + utils.Colors.GREEN
        + "BARZILAI-BORWEIN GRADIENT METHOD\t\t\tTolerance = {:.1e}"
        + utils.Colors.ENDC)
    print(utils.Colors.UNDERLINE + ' '*80 + utils.Colors.ENDC)
    print(string.format(ocp.param.ocptol))
    print(utils.Colors.UNDERLINE + ' '*80 + utils.Colors.ENDC + "\n")

    # Initialize list of cost values, norm of optimality residual and relative error
    COST_VALUES = []
    OPT_REST_NORM = []
    REL_ERROR = []

    # main algorithm
    start = time()
    for i in range(ocp.param.ocpmaxit):
        if i == 0:
            print("Iteration: 0")
            state = ocp.state_solver(ocp.init_control)
            residue = ocp.residual(state)
            cost_old = ocp.cost(residue, state, ocp.init_control)
            adjoint = ocp.adjoint_solver(state, residue)
            control_old = ocp.init_control
            control = ocp.init_control \
                - ocp.der_cost(ocp.init_control,
                ocp.adjoint_to_control(adjoint))
            print("\nIteration: 1")
            state = ocp.state_solver(control)
            residue = ocp.residual(state)
            cost = ocp.cost(residue, state, control)
            steplength = 1.0
            rel_error = np.abs(cost - cost_old) / cost
            opt_res = ocp.der_cost(control, ocp.adjoint_to_control(adjoint))
            opt_res_norm = ocp.der_cost_norm(opt_res)
            string = ("\n\tCost Value = {:.8e}\tRelative Error = {:.8e}")
            print(string.format(cost, rel_error))
            string = ("\tSteplength = {:.8e}\tOptimality Res = {:.8e}")
            print(string.format(steplength, opt_res_norm))
            COST_VALUES += [cost]
            OPT_REST_NORM += [opt_res_norm]
            REL_ERROR += [rel_error]
        else:
            print("\nIteration: {}".format(i+1))
            adjoint_old = ocp.adjoint_to_control(adjoint)
            adjoint = ocp.adjoint_solver(state, residue)
            control_residue = control - control_old
            adjoint_residue = ocp.adjoint_to_control(adjoint) - adjoint_old
            res_dercost = ocp.der_cost(control_residue, adjoint_residue)
            if version == 1:
                steplength = np.sum(control_residue * res_dercost) \
                             / np.sum(res_dercost * res_dercost)
            elif version == 2:
                steplength = np.sum(control_residue * control_residue) \
                             / np.sum(control_residue * res_dercost)
            elif version == 3:
                if (i % 2) == 1:
                    steplength = np.sum(control_residue * res_dercost) \
                                 / np.sum(res_dercost * res_dercost)
                else:
                    steplength = np.sum(control_residue * control_residue) \
                                 / np.sum(control_residue * res_dercost)
            control_old = control
            opt_res = ocp.der_cost(control, ocp.adjoint_to_control(adjoint))
            control = control - steplength * opt_res
            state = ocp.state_solver(control)
            cost_old = cost
            residue = ocp.residual(state)
            cost = ocp.cost(residue, state, control)
            rel_error = np.abs(cost - cost_old) / cost
            opt_res = ocp.der_cost(control, ocp.adjoint_to_control(adjoint))
            opt_res_norm = ocp.der_cost_norm(opt_res)
            string = ("\n\tCost Value = {:.8e}\tRelative Error = {:.8e}")
            print(string.format(cost, rel_error))
            string = ("\tSteplength = {:.8e}\tOptimality Res = {:.8e}")
            print(string.format(steplength, opt_res_norm))
            COST_VALUES += [cost]
            OPT_REST_NORM += [opt_res_norm]
            REL_ERROR += [rel_error]

            if max(opt_res_norm, rel_error) < ocp.param.ocptol:
                print(utils.Colors.BOLD + utils.Colors.GREEN
                    + "\nOptimal solution found.")
                break
            if i == ocp.param.ocpmaxit - 1 \
                and max(opt_res_norm, rel_error) < ocp.param.ocptol:
                print("BB Warning: Maximum number of iterations reached"
                      "without satisfying the tolerance.")
    end = time()
    print("\tElapsed time is {:.8f} seconds.".format(end-start)
        + utils.Colors.ENDC)
    print(utils.Colors.UNDERLINE + ' '*80 + utils.Colors.ENDC + "\n")

    return state, adjoint, control, residue, COST_VALUES, OPT_REST_NORM, REL_ERROR

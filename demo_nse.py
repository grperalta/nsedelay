# -*- coding: utf-8 -*-
"""
This python script runs test for the implicit-explicit (IMEX) Euler scheme
for the Navier-Stokes equation using the Taylor-Hood and Mini-Finite elements.

Gilbert Peralta
Department of Mathematics and Computer Science
University of the Philippines Baguio
Governor Pack Road, Baguio, Philippines 2600
Email: grperalta@up.edu.ph
9 July 2020
"""

from __future__ import division
from scipy import sparse as sp
from scipy.sparse.linalg import spsolve
from demo_stokes import print_order_convergence
import stokesfem as sfem
import functions as fun
import numpy as np
import utils

__author__ = "Gilbert Peralta"
__institution__ = "University of the Philippines Baguio"
__email__ = "grperalta@up.edu.ph"
__date__ = "9 July 2020"


def get_nse_fem_matrices(mesh, femdatastruct, dt):
    """
    Returns the total matrix, mass matrix, pressure mass matrix and indices
    of Dirichlet nodes.

    Parameters
    ----------
    mesh : stokesfem.Mesh class
        Triangulation of the domain.
    femdatastruct : stokesfem.FEMDataStruct class
        Finite element data structure.
    dt : float
        Temporal grid size.
    """

    # matrix assembly
    K, M, Bx, By, Mp = sfem.assemble(mesh, femdatastruct)

    # Total matrix
    Atotal = (1.0 / dt) * M + fun.VISCO * K
    Atotal = sp.kron(sp.identity(2), Atotal).tocsr()
    B = sp.bmat([[Bx, By]], format='csr')
    Atotal = Atotal + 1e10 * (B.T * B)

    # Dirichlet nodes
    if femdatastruct.name is 'taylor_hood':
        dirichlet_bc \
            = np.append(mesh.all_bdy_node(), mesh.dof + mesh.all_bdy_node())
    else:
        dirichlet_bc = np.append(mesh.bdy_node, mesh.dof + mesh.bdy_node)

    # Apply no-slip boundary condition to total matrix
    Atotal = sfem.apply_noslip_bc(Atotal, dirichlet_bc)

    # sparse LU factorization
    splu_opts = dict(DiagPivotThresh=1e-6, SymmetricMode=True,
        PivotGrowth=True)
    Asolve = sp.linalg.splu(Atotal, permc_spec="MMD_AT_PLUS_A",
        panel_size=3, options=splu_opts).solve

    return Asolve, M, Mp, dirichlet_bc, B


def L2_spacetime_error(M, dt, u, v):
    """
    Computes the space-time L2-error norm of velocity field U = (u, v).

    Parameters
    ----------
    M : scipy.sparse.csr_matrix
        Mass matrix.
    dt : float
        Temporal grid size.
    u, v : numpy.ndarray
        Components of the velocity vector field V = (u, v).
    """
    L2_error2 = np.sum(u * (M * u)) + np.sum(v * (M * v))

    return np.sqrt(dt * L2_error2)


def navier_stokes(name='taylor_hood'):
    """
    Test problem for the IMEX Euler scheme for NSE.

    Parameter
    ---------
    name : str
        Either 'taylor_hood' or 'p1_bubble'.
    """
    print("\nIMEX EULER SCHEME FOR NAVIER-STOKES EQUATION VIA {} FEM".format(
        name.upper()))
    list_Nx, list_Nt, velocity_error, pressure_error, mesh_sizes \
        = [9, 17, 33], [11, 41, 161], [], [], []

    for i in range(len(list_Nx)):
        # temporal mesh
        t = np.linspace(0, 1, list_Nt[i])
        dt = t[1] - t[0]

        # mesh generation
        mesh = sfem.square_uniform_trimesh(list_Nx[i]).femprocess(name=name)

        # exact data
        if name is 'taylor_hood':
            u_exact = fun.U_EXACT(t, mesh.all_node()[:, 0], mesh.all_node()[:, 1])
            v_exact = fun.V_EXACT(t, mesh.all_node()[:, 0], mesh.all_node()[:, 1])
            force_x = fun.F_EXACT(t, mesh.all_node()[:, 0], mesh.all_node()[:, 1])
            force_y = fun.G_EXACT(t, mesh.all_node()[:, 0], mesh.all_node()[:, 1])
        else:
            u_exact = fun.U_EXACT_BUBBLE(t, mesh)
            v_exact = fun.V_EXACT_BUBBLE(t, mesh)
            force_x = fun.F_EXACT_BUBBLE(t, mesh)
            force_y = fun.G_EXACT_BUBBLE(t, mesh)
        p_exact = fun.P_EXACT(t[1:], mesh.node[:, 0], mesh.node[:, 1])

        # fem data structure
        femdatastruct = sfem.get_fem_data_struct(mesh, name=name)

        # matrix assembly
        Asolve, M, Mp, dirichlet_bc, B \
            = get_nse_fem_matrices(mesh, femdatastruct, dt)

        # pre-allocation of solution vectors
        u_app = np.zeros(u_exact.shape).astype(float)
        v_app = np.zeros(v_exact.shape).astype(float)
        p_app = np.zeros(p_exact.shape).astype(float)

        # initial data
        u_app[:, 0] = u_exact[:, 0]
        v_app[:, 0] = v_exact[:, 0]

        for it in range(list_Nt[i]-1):
            # convection matrix
            N = sfem.convection(mesh, u_app[:, it], v_app[:, it], femdatastruct)

            # right hand vector
            rhs = np.append(
                N * u_app[:, it] + M * ((1/dt)*u_app[:, it] + force_x[:, it+1]),
                N * v_app[:, it] + M * ((1/dt)*v_app[:, it] + force_y[:, it+1]))
            rhs[dirichlet_bc] = 0.0

            # update velocity and pressure
            u_app[:, it+1], v_app[:, it+1] = np.split(Asolve(rhs), 2)
            p_app[:, it] = - 1e10 * B * np.append(u_app[:, it+1], v_app[:, it+1])

        mesh_sizes += [mesh.size()]
        velocity_error \
            += [L2_spacetime_error(M, dt, u_app - u_exact, v_app - v_exact)]
        dp = p_app - p_exact
        pressure_error += [np.sqrt(dt * np.sum(dp * (Mp * dp)))]

    print("")
    print_order_convergence(mesh_sizes, velocity_error, pressure_error)


if __name__ == '__main__':
    utils.print_start_implementation()
    navier_stokes('p1_bubble')
    navier_stokes('taylor_hood')
    utils.print_end_implementation()

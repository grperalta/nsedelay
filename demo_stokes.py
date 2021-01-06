# -*- coding: utf-8 -*-
"""
This python script runs tests for the matrix assemblies in the module
stokesfem.py

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
import numpy as np
import stokesfem as sfem
import functions as fun
import utils

__author__ = "Gilbert Peralta"
__institution__ = "University of the Philippines Baguio"
__email__ = "grperalta@up.edu.ph"
__date__ = "9 July 2020"

def matrix_assembly(mesh, femdatastruct):
    """
    Returns the total matrix, mass matrix, pressure mass matrix and indices
    of Dirichlet nodes.

    Parameters
    ----------
    mesh : stokesfem.Mesh class
        Triangulation of the domain.
    femdatastruct : stokesfem.FEMDataStruct class
        Finite element data structure.
    """
    # matrix assembly
    K, M, Bx, By, Mp = sfem.assemble(mesh, femdatastruct)

    # penalty matrix
    Z = sp.spdiags(1e-10*np.ones(mesh.num_node), 0, mesh.num_node, mesh.num_node)

    # total matrix
    Atotal = sp.bmat([[K, None, -Bx.T],
                     [None, K, -By.T],
                     [Bx, By, Z]], format='csr')

    # Dirichlet nodes
    if femdatastruct.name is 'taylor_hood':
        dirichlet_bc \
            = np.append(mesh.all_bdy_node(), mesh.dof + mesh.all_bdy_node())
    else:
        dirichlet_bc = np.append(mesh.bdy_node, mesh.dof + mesh.bdy_node)

    # Apply no-slip boundary condition to total matrix
    Atotal = sfem.apply_noslip_bc(Atotal, dirichlet_bc)

    return Atotal, M, Mp, dirichlet_bc


def L2_error(M, u, v):
    """
    Computes the L2-error norm of velocity field U = (u, v).

    Parameters
    ----------
    M : scipy.sparse.csr_matrix
        Mass matrix.
    u, v : numpy.ndarray
        Components of the velocity vector field V = (u, v).
    """
    return np.sqrt(np.dot(M * u, u) + np.dot(M * v, v))


def print_order_convergence(mesh_sizes, error_velocity, error_pressure):
    """
    Table for the mesh sizes, L2-error norms and order of convergences.

    Parameters
    ----------
    mesh_sizes : list
        List of triangular mesh sizes.
    error_velocity : list
        List of L2-error norms for velocity.
    error_pressure : list
        List of L2-error norms for pressure.
    """
    print("="*80)
    print("Mesh Size\t\tVelocity Error\t\t\tOrder of Convergence")
    print("="*80 + utils.Colors.BOLD + utils.Colors.GREEN)

    # velocity errors
    for k in range(len(mesh_sizes)):
        if k == 0:
            print("{:.16f}\t{:.16e}".format(
                mesh_sizes[k], error_velocity[k]))
        else:
            eoc = np.log(error_velocity[k] / error_velocity[k-1]) \
                / np.log(mesh_sizes[k] / mesh_sizes[k-1])
            print("{:.16f}\t{:.16e}\t\t{:.16f}".format(
                mesh_sizes[k], error_velocity[k], eoc))

    print(utils.Colors.ENDC + "="*80)
    print("Mesh Size\t\tPressure Error\t\t\tOrder of Convergence")
    print("="*80 + utils.Colors.BOLD + utils.Colors.GREEN)

    # pressure errors
    for k in range(len(mesh_sizes)):
        if k == 0:
            print("{:.16f}\t{:.16e}".format(
                mesh_sizes[k], error_velocity[k]))
        else:
            eoc = np.log(error_pressure[k] / error_pressure[k-1]) \
                / np.log(mesh_sizes[k] / mesh_sizes[k-1])
            print("{:.16f}\t{:.16e}\t\t{:.16f}".format(
                mesh_sizes[k], error_velocity[k], eoc))

    print(utils.Colors.ENDC + "="*80 + "\n")


def stokes_taylorhood():
    """
    TEST PROBLEM: TAYLOR-HOOD FEM FOR THE BOUNDARY VALUE PROBLEM:
            - Delta U + Grad p = F in Omega,
                         Div U = 0 in Omega,
                             U = 0 in Gamma.
    """
    print(utils.Colors.BOLD + utils.Colors.BLUE
        + "\nTEST PROBLEM: TAYLOR-HOOD FEM FOR THE BOUNDARY VALUE PROBLEM: \n"
        + "\t\t - Delta U + Grad p = F in Omega, \n"
        + "\t\t              Div U = 0 in Omega, \n"
        + "\t\t                  U = 0 in Gamma. \n"
        + utils.Colors.ENDC)

    subdiv_list, error_velocity, error_pressure, mesh_sizes \
        = [11, 21, 31, 41], [], [], []
    ctr = 1

    for n in subdiv_list:
        # mesh generation
        mesh = sfem.square_uniform_trimesh(n).femprocess()
        print("Triangulation {}".format(ctr))
        print("\t" + str(mesh) + "\n")

        # fem data structure
        femdatastruct = sfem.get_fem_data_struct(mesh)

        Atotal, M, Mp, dirichlet_bc = matrix_assembly(mesh, femdatastruct)

        # exact solutions and external force
        mesh.all_node = mesh.all_node()
        u_exact = fun.u(mesh.all_node[:, 0], mesh.all_node[:, 1])
        v_exact = fun.v(mesh.all_node[:, 0], mesh.all_node[:, 1])
        force_x = fun.f(mesh.all_node[:, 0], mesh.all_node[:, 1])
        force_y = fun.g(mesh.all_node[:, 0], mesh.all_node[:, 1])
        p_exact = fun.p(mesh.node[:, 0], mesh.node[:, 1])

        # right hand side vector
        rhs = np.hstack([M * force_x, M * force_y, np.zeros(mesh.num_node)])
        rhs[dirichlet_bc] = 0.0

        # solve linear system
        sol = spsolve(Atotal, rhs)
        u_app, v_app, p_app \
            = sol[:mesh.dof], sol[mesh.dof:2*mesh.dof], sol[2*mesh.dof:]

        mesh_sizes += [mesh.size()]
        error_velocity \
            += [L2_error(M, u_app - u_exact, v_app - v_exact)]
        error_pressure \
            += [np.sqrt(np.dot(p_app - p_exact, Mp * (p_app - p_exact)))]
        ctr += 1

    print_order_convergence(mesh_sizes, error_velocity, error_pressure)


def stokes_bubble():
    """
    TEST PROBLEM: P1-BUBBLE FEM FOR THE BOUNDARY VALUE PROBLEM:
            - Delta U + Grad p = F in Omega,
                         Div U = 0 in Omega,
                             U = 0 in Gamma.
    """
    print(utils.Colors.BOLD + utils.Colors.BLUE
        + "\nTEST PROBLEM: P1-BUBBLE FEM FOR THE BOUNDARY VALUE PROBLEM: \n"
        + "\t\t - Delta U + Grad p = F in Omega, \n"
        + "\t\t              Div U = 0 in Omega, \n"
        + "\t\t                  U = 0 in Gamma. \n"
        + utils.Colors.ENDC)

    subdiv_list, error_velocity, error_pressure, mesh_sizes \
        = [11, 21, 31, 41], [], [], []
    ctr = 1

    for n in subdiv_list:
        # mesh generation
        mesh = sfem.square_uniform_trimesh(n).femprocess('p1_bubble')
        print("Triangulation {}".format(ctr))
        print("\t>>> " + str(mesh) + "\n")

        # fem data structure
        femdatastruct = sfem.get_fem_data_struct(mesh, name='p1_bubble')

        Atotal, M, Mp, dirichlet_bc = matrix_assembly(mesh, femdatastruct)

        # exact solutions and external force
        u_exact = sfem.bubble_interpolation(mesh, fun.u)
        v_exact = sfem.bubble_interpolation(mesh, fun.v)
        p_exact = fun.p(mesh.node[:, 0], mesh.node[:, 1])
        force_x = sfem.bubble_interpolation(mesh, fun.f)
        force_y = sfem.bubble_interpolation(mesh, fun.g)

        # right hand side vector
        rhs = np.hstack([M * force_x, M * force_y, np.zeros(mesh.num_node)])
        rhs[dirichlet_bc] = 0.0

        # solve linear system
        sol = spsolve(Atotal, rhs)
        u_app, v_app, p_app \
            = sol[:mesh.dof], sol[mesh.dof:2*mesh.dof], sol[2*mesh.dof:]

        mesh_sizes += [mesh.size()]
        error_velocity \
            += [L2_error(M, u_app - u_exact, v_app - v_exact)]
        error_pressure \
            += [np.sqrt(np.dot(p_app - p_exact, Mp * (p_app - p_exact)))]
        ctr += 1

    print_order_convergence(mesh_sizes, error_velocity, error_pressure)


def stokes_convection_taylorhood():
    """
    TEST PROBLEM: TAYLOR-HOOD FEM FOR THE BOUNDARY VALUE PROBLEM:
            - Delta U + Grad p = F + (U.Grad) U in Omega,
                         Div U = 0              in Omega,
                             U = 0              in Gamma.
    """
    print(utils.Colors.BOLD + utils.Colors.BLUE
        + "\nTEST PROBLEM: TAYLOR-HOOD FEM FOR THE BOUNDARY VALUE PROBLEM: \n"
        + "\t\t - Delta U + Grad p = F + (U.Grad) U in Omega, \n"
        + "\t\t              Div U = 0              in Omega, \n"
        + "\t\t                  U = 0              in Gamma. \n"
        + utils.Colors.ENDC)

    subdiv_list, error_velocity, error_pressure, mesh_sizes \
        = [11, 21, 31, 41], [], [], []
    ctr = 1

    for n in subdiv_list:
        # mesh generation
        mesh = sfem.square_uniform_trimesh(n).femprocess()
        print("Triangulation {}".format(ctr))
        print("\t>>> " + str(mesh) + "\n")

        # fem data structure
        femdatastruct = sfem.get_fem_data_struct(mesh)

        Atotal, M, Mp, dirichlet_bc = matrix_assembly(mesh, femdatastruct)

        # exact solutions and external force
        mesh.all_node = mesh.all_node()
        u_exact = fun.u(mesh.all_node[:, 0], mesh.all_node[:, 1])
        v_exact = fun.v(mesh.all_node[:, 0], mesh.all_node[:, 1])
        p_exact = fun.p(mesh.node[:, 0], mesh.node[:, 1])
        force_x = fun.f(mesh.all_node[:, 0], mesh.all_node[:, 1]) \
            + u_exact * fun.u_x(mesh.all_node[:, 0], mesh.all_node[:, 1]) \
            + v_exact * fun.u_y(mesh.all_node[:, 0], mesh.all_node[:, 1])
        force_y = fun.g(mesh.all_node[:, 0], mesh.all_node[:, 1]) \
            + u_exact * fun.v_x(mesh.all_node[:, 0], mesh.all_node[:, 1]) \
            + v_exact * fun.v_y(mesh.all_node[:, 0], mesh.all_node[:, 1])

        # assembly of convection matrix
        N = sfem.convection(mesh, u_exact, v_exact, femdatastruct)

        # right hand side vector
        rhs = np.hstack([M * force_x + N * u_exact, M * force_y + N * v_exact,
            np.zeros(mesh.num_node)])
        rhs[dirichlet_bc] = 0.0

        # solve linear system
        sol = spsolve(Atotal, rhs)
        u_app, v_app, p_app \
            = sol[:mesh.dof], sol[mesh.dof:2*mesh.dof], sol[2*mesh.dof:]

        mesh_sizes += [mesh.size()]
        error_velocity \
            += [L2_error(M, u_app - u_exact, v_app - v_exact)]
        error_pressure \
            += [np.sqrt(np.dot(p_app - p_exact, Mp * (p_app - p_exact)))]
        ctr += 1

    print_order_convergence(mesh_sizes, error_velocity, error_pressure)


def stokes_convection_bubble():
    """
    TEST PROBLEM: P1-BUBBLE FEM FOR THE BOUNDARY VALUE PROBLEM:
            - Delta U + Grad p = F + (U.Grad) U in Omega,
                         Div U = 0              in Omega,
                             U = 0              in Gamma.
    """
    print(utils.Colors.BOLD + utils.Colors.BLUE
        + "\nTEST PROBLEM: P1-BUBBLE FEM FOR THE BOUNDARY VALUE PROBLEM: \n"
        + "\t\t - Delta U + Grad p = F + (U.Grad) U in Omega, \n"
        + "\t\t              Div U = 0              in Omega, \n"
        + "\t\t                  U = 0              in Gamma. \n"
        + utils.Colors.ENDC)

    subdiv_list, error_velocity, error_pressure, mesh_sizes \
        = [11, 21, 31, 41], [], [], []
    ctr = 1

    for n in subdiv_list:
        # mesh generation
        mesh = sfem.square_uniform_trimesh(n).femprocess('p1_bubble')
        print("Triangulation {}".format(ctr))
        print("\t>>> " + str(mesh) + "\n")

        # fem data structure
        femdatastruct = sfem.get_fem_data_struct(mesh, name='p1_bubble')

        Atotal, M, Mp, dirichlet_bc = matrix_assembly(mesh, femdatastruct)

        # exact solutions and external force
        u_exact = sfem.bubble_interpolation(mesh, fun.u)
        v_exact = sfem.bubble_interpolation(mesh, fun.v)
        p_exact = fun.p(mesh.node[:, 0], mesh.node[:, 1])
        force_x = sfem.bubble_interpolation(mesh, fun.f) \
            + u_exact * sfem.bubble_interpolation(mesh, fun.u_x) \
            + v_exact * sfem.bubble_interpolation(mesh, fun.u_y)
        force_y = sfem.bubble_interpolation(mesh, fun.g) \
            + u_exact * sfem.bubble_interpolation(mesh, fun.v_x) \
            + v_exact * sfem.bubble_interpolation(mesh, fun.v_y)

        # assembly of convection matrix
        N = sfem.convection(mesh, u_exact, v_exact, femdatastruct)

        # right hand side vector
        rhs = np.hstack([M * force_x + N * u_exact, M * force_y + N * v_exact,
            np.zeros(mesh.num_node)])
        rhs[dirichlet_bc] = 0.0

        # solve linear system
        sol = spsolve(Atotal, rhs)
        u_app, v_app, p_app \
            = sol[:mesh.dof], sol[mesh.dof:2*mesh.dof], sol[2*mesh.dof:]

        mesh_sizes += [mesh.size()]
        error_velocity \
            += [L2_error(M, u_app - u_exact, v_app - v_exact)]
        error_pressure \
            += [np.sqrt(np.dot(p_app - p_exact, Mp * (p_app - p_exact)))]
        ctr += 1

    print_order_convergence(mesh_sizes, error_velocity, error_pressure)


def stokes_convection_dual_taylorhood():
    """
    TEST PROBLEM: TAYLOR-HOOD FEM FOR THE BOUNDARY VALUE PROBLEM:
            - Delta U + Grad p = F + (U.Grad).T Grad p in Omega,
                         Div U = 0                     in Omega,
                             U = 0                     in Gamma.
    """
    print(utils.Colors.BOLD + utils.Colors.BLUE
        + "\nTEST PROBLEM: TAYLOR-HOOD FEM FOR THE BOUNDARY VALUE PROBLEM: \n"
        + "\t\t - Delta U + Grad p = F + (Grad U).T Grad p in Omega, \n"
        + "\t\t              Div U = 0                     in Omega, \n"
        + "\t\t                  U = 0                     in Gamma. \n"
        + utils.Colors.ENDC)

    subdiv_list, error_velocity, error_pressure, mesh_sizes \
        = [11, 21, 31, 41], [], [], []
    ctr = 1

    for n in subdiv_list:
        # mesh generation
        mesh = sfem.square_uniform_trimesh(n).femprocess()
        print("Triangulation {}".format(ctr))
        print("\t>>> " + str(mesh) + "\n")

        # fem data structure
        femdatastruct = sfem.get_fem_data_struct(mesh)

        Atotal, M, Mp, dirichlet_bc = matrix_assembly(mesh, femdatastruct)

        # exact solutions and external force
        mesh.all_node = mesh.all_node()
        u_exact = fun.u(mesh.all_node[:, 0], mesh.all_node[:, 1])
        v_exact = fun.v(mesh.all_node[:, 0], mesh.all_node[:, 1])
        p_exact = fun.p(mesh.node[:, 0], mesh.node[:, 1])
        ux_exact = fun.u_x(mesh.all_node[:, 0], mesh.all_node[:, 1])
        uy_exact = fun.u_y(mesh.all_node[:, 0], mesh.all_node[:, 1])
        vx_exact = fun.v_x(mesh.all_node[:, 0], mesh.all_node[:, 1])
        vy_exact = fun.v_y(mesh.all_node[:, 0], mesh.all_node[:, 1])
        px_exact = fun.p_x(mesh.all_node[:, 0], mesh.all_node[:, 1])
        py_exact = fun.p_y(mesh.all_node[:, 0], mesh.all_node[:, 1])
        force_x = fun.f(mesh.all_node[:, 0], mesh.all_node[:, 1]) \
            + ux_exact * px_exact + uy_exact * py_exact
        force_y = fun.g(mesh.all_node[:, 0], mesh.all_node[:, 1]) \
            + vx_exact * px_exact + vy_exact * py_exact

        # assembly of dual convection matrix
        N = sfem.convection_dual(mesh, u_exact, v_exact, femdatastruct)

        # right hand side vector
        rhs = np.hstack([M * force_x, M * force_y]) \
            - N * np.append(px_exact, py_exact)
        rhs = np.hstack([rhs, np.zeros(mesh.num_node)])
        rhs[dirichlet_bc] = 0.0

        # solve linear system
        sol = spsolve(Atotal, rhs)
        u_app, v_app, p_app \
            = sol[:mesh.dof], sol[mesh.dof:2*mesh.dof], sol[2*mesh.dof:]

        mesh_sizes += [mesh.size()]
        error_velocity \
            += [L2_error(M, u_app - u_exact, v_app - v_exact)]
        error_pressure \
            += [np.sqrt(np.dot(p_app - p_exact, Mp * (p_app - p_exact)))]
        ctr += 1

    print_order_convergence(mesh_sizes, error_velocity, error_pressure)


def stokes_convection_dual_bubble():
    """
    TEST PROBLEM: P1-BUBBLE FEM FOR THE BOUNDARY VALUE PROBLEM:
            - Delta U + Grad p = F + (U.Grad).T Grad p in Omega,
                         Div U = 0                     in Omega,
                             U = 0                     in Gamma.
    """
    print(utils.Colors.BOLD + utils.Colors.BLUE
        + "\nTEST PROBLEM: P1-BUBBLE FEM FOR THE BOUNDARY VALUE PROBLEM: \n"
        + "\t\t - Delta U + Grad p = F + (Grad U).T Grad p in Omega, \n"
        + "\t\t              Div U = 0                     in Omega, \n"
        + "\t\t                  U = 0                     in Gamma. \n"
        + utils.Colors.ENDC)

    subdiv_list, error_velocity, error_pressure, mesh_sizes \
        = [11, 21, 31, 41], [], [], []
    ctr = 1

    for n in subdiv_list:
        # mesh generation
        mesh = sfem.square_uniform_trimesh(n).femprocess('p1_bubble')
        print("Triangulation {}".format(ctr))
        print("\t>>> " + str(mesh) + "\n")

        # fem data structure
        femdatastruct = sfem.get_fem_data_struct(mesh, name='p1_bubble')

        Atotal, M, Mp, dirichlet_bc = matrix_assembly(mesh, femdatastruct)

        # exact solutions and external force
        u_exact = sfem.bubble_interpolation(mesh, fun.u)
        v_exact = sfem.bubble_interpolation(mesh, fun.v)
        p_exact = fun.p(mesh.node[:, 0], mesh.node[:, 1])
        px_exact = sfem.bubble_interpolation(mesh, fun.p_x)
        py_exact = sfem.bubble_interpolation(mesh, fun.p_y)
        force_x = sfem.bubble_interpolation(mesh, fun.f) \
            + sfem.bubble_interpolation(mesh, fun.u_x) * px_exact \
            + sfem.bubble_interpolation(mesh, fun.u_y) * py_exact
        force_y = sfem.bubble_interpolation(mesh, fun.g) \
            + sfem.bubble_interpolation(mesh, fun.v_x) * px_exact \
            + sfem.bubble_interpolation(mesh, fun.v_y) * py_exact

        # assembly of dual convection matrix
        N = sfem.convection_dual(mesh, u_exact, v_exact, femdatastruct)

        # right hand side vector
        rhs = np.hstack([M * force_x, M * force_y]) \
            - N * np.append(px_exact, py_exact)
        rhs = np.hstack([rhs, np.zeros(mesh.num_node)])
        rhs[dirichlet_bc] = 0.0

        # solve linear system
        sol = spsolve(Atotal, rhs)
        u_app, v_app, p_app \
            = sol[:mesh.dof], sol[mesh.dof:2*mesh.dof], sol[2*mesh.dof:]

        mesh_sizes += [mesh.size()]
        error_velocity \
            += [L2_error(M, u_app - u_exact, v_app - v_exact)]
        error_pressure \
            += [np.sqrt(np.dot(p_app - p_exact, Mp * (p_app - p_exact)))]
        ctr += 1

    print_order_convergence(mesh_sizes, error_velocity, error_pressure)


def stokes_vorticity_taylorhood():
    """
    TEST PROBLEM: TAYLOR-HOOD FEM FOR THE BOUNDARY VALUE PROBLEM:
            - Delta U + Grad p = F + Curl.T Curl U in Omega,
                         Div U = 0                 in Omega,
                             U = 0                 in Gamma.
    """
    print(utils.Colors.BOLD + utils.Colors.BLUE
        + "\nTEST PROBLEM: TAYLOR-HOOD FEM FOR THE BOUNDARY VALUE PROBLEM: \n"
        + "\t\t - Delta U + Grad p = F + Curl.T Curl U in Omega, \n"
        + "\t\t              Div U = 0                 in Omega, \n"
        + "\t\t                  U = 0                 in Gamma. \n"
        + utils.Colors.ENDC)

    subdiv_list, error_velocity, error_pressure, mesh_sizes \
        = [11, 21, 31, 41], [], [], []
    ctr = 1

    for n in subdiv_list:
        # mesh generation
        mesh = sfem.square_uniform_trimesh(n).femprocess()
        print("\t>>> " + str(mesh) + "\n")

        # fem data structure
        femdatastruct = sfem.get_fem_data_struct(mesh)

        Atotal, M, Mp, dirichlet_bc = matrix_assembly(mesh, femdatastruct)

        # exact solutions and external force
        mesh.all_node = mesh.all_node()
        u_exact = fun.u(mesh.all_node[:, 0], mesh.all_node[:, 1])
        v_exact = fun.v(mesh.all_node[:, 0], mesh.all_node[:, 1])
        p_exact = fun.p(mesh.node[:, 0], mesh.node[:, 1])
        force_x = fun.f(mesh.all_node[:, 0], mesh.all_node[:, 1]) \
            + fun.U_curlcurl_x(mesh.all_node[:, 0], mesh.all_node[:, 1])
        force_y = fun.g(mesh.all_node[:, 0], mesh.all_node[:, 1]) \
            + fun.U_curlcurl_y(mesh.all_node[:, 0], mesh.all_node[:, 1])

        # assembly of voriticity matrix
        V = sfem.vorticity(mesh, femdatastruct)

        # right hand side vector
        rhs = np.hstack([M * force_x, M * force_y]) \
            - V * np.append(u_exact, v_exact)
        rhs = np.hstack([rhs, np.zeros(mesh.num_node)])
        rhs[dirichlet_bc] = 0.0

        # solve linear system
        sol = spsolve(Atotal, rhs)
        u_app, v_app, p_app \
            = sol[:mesh.dof], sol[mesh.dof:2*mesh.dof], sol[2*mesh.dof:]

        mesh_sizes += [mesh.size()]
        error_velocity \
            += [L2_error(M, u_app - u_exact, v_app - v_exact)]
        error_pressure \
            += [np.sqrt(np.dot(p_app - p_exact, Mp * (p_app - p_exact)))]
        ctr += 1

    print_order_convergence(mesh_sizes, error_velocity, error_pressure)


def stokes_vorticity_bubble():
    """
    TEST PROBLEM: P1-BUBBLE FEM FOR THE BOUNDARY VALUE PROBLEM:
            - Delta U + Grad p = F + Curl.T Curl U in Omega,
                         Div U = 0                 in Omega,
                             U = 0                 in Gamma.
    """
    print(utils.Colors.BOLD + utils.Colors.BLUE
        + "\nTEST PROBLEM: P1-BUBBLE FEM FOR THE BOUNDARY VALUE PROBLEM: \n"
        + "\t\t - Delta U + Grad p = F + Curl.T Curl U in Omega, \n"
        + "\t\t              Div U = 0                 in Omega, \n"
        + "\t\t                  U = 0                 in Gamma. \n"
        + utils.Colors.ENDC)

    subdiv_list, error_velocity, error_pressure, mesh_sizes \
        = [11, 21, 31, 41], [], [], []
    ctr = 1

    for n in subdiv_list:
        # mesh generation
        mesh = sfem.square_uniform_trimesh(n).femprocess('p1_bubble')
        print("Triangulation {}".format(ctr))
        print("\t>>> " + str(mesh) + "\n")

        # fem data structure
        femdatastruct = sfem.get_fem_data_struct(mesh, name='p1_bubble')

        Atotal, M, Mp, dirichlet_bc = matrix_assembly(mesh, femdatastruct)

        # exact solutions and external force
        u_exact = sfem.bubble_interpolation(mesh, fun.u)
        v_exact = sfem.bubble_interpolation(mesh, fun.v)
        p_exact = fun.p(mesh.node[:, 0], mesh.node[:, 1])
        force_x = sfem.bubble_interpolation(mesh, fun.f) \
            + sfem.bubble_interpolation(mesh, fun.U_curlcurl_x)
        force_y = sfem.bubble_interpolation(mesh, fun.g) \
            + sfem.bubble_interpolation(mesh, fun.U_curlcurl_y)

        # assembly of vorticity matrix
        V = sfem.vorticity(mesh, femdatastruct)

        # right hand side vector
        rhs = np.hstack([M * force_x, M * force_y]) \
            - V * np.append(u_exact, v_exact)
        rhs = np.hstack([rhs, np.zeros(mesh.num_node)])
        rhs[dirichlet_bc] = 0.0

        # solve linear system
        sol = spsolve(Atotal, rhs)
        u_app, v_app, p_app \
            = sol[:mesh.dof], sol[mesh.dof:2*mesh.dof], sol[2*mesh.dof:]

        mesh_sizes += [mesh.size()]
        error_velocity \
            += [L2_error(M, u_app - u_exact, v_app - v_exact)]
        error_pressure \
            += [np.sqrt(np.dot(p_app - p_exact, Mp * (p_app - p_exact)))]
        ctr += 1

    print_order_convergence(mesh_sizes, error_velocity, error_pressure)


if  __name__ == '__main__':
    utils.print_start_implementation()
    stokes_taylorhood()
    stokes_bubble()
    stokes_convection_taylorhood()
    stokes_convection_bubble()
    stokes_convection_dual_bubble()
    stokes_vorticity_taylorhood()
    stokes_vorticity_bubble()
    utils.print_end_implementation()

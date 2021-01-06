# -*- coding: utf-8 -*-
"""
This python script implements the velocity tracking problem for the optimal
control of the delayed Navier-Stokes equation using the implicit-explicit
(IMEX) Euler scheme.

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
from matplotlib import cm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import nsedelay as nse
import stokesfem as sfem
import utils
import os

__author__ = "Gilbert Peralta"
__institution__ = "University of the Philippines Baguio"
__email__ = "grperalta@up.edu.ph"
__date__ = "7 January 2021"


def check_dir():
    """
    Check if there is a directory 'npyfiles_velocity', otherwise, create one.
    """
    if os.path.isdir('npyfiles_velocity'):
        pass
    else:
        os.mkdir('npyfiles_velocity')


def get_null_control_nodes(mesh, XLIM, YLIM):
    """
    Returns the list of node indices where no control is applied. Here, mesh is a
    stokesfem.Mesh class for the domain triangulation, XLIM and YLIM are lists
    corresponding to the bounding box for the control region.
    """
    control_node = []
    mesh_all_node = mesh.all_node()
    for i in range(mesh.dof):
        if mesh_all_node[i, 0] >= XLIM[0] and mesh_all_node[i, 0] <= XLIM[1]:
            if mesh_all_node[i, 1] >= YLIM[0] and mesh_all_node[i, 1] <= YLIM[1]:
                control_node += [i]
    null_control = np.array(list(set(range(mesh.dof)).difference(control_node)))
    null_control = np.append(null_control, null_control + mesh.dof)

    return list(null_control)


def get_ocp(NX, NY, tau, XLIM, YLIM, femtype='p1_bubble'):
    """
    Returns nsedelay.OCP class for the optimal control problem.
    """
    # parameters
    param = nse.Parameters.get_default()
    param.nu = 0.01
    param.Nt = 101
    param.tau = tau
    param.ocptol = 1e-5
    param.alpha = 1e-3
    param.alpha_O = 1.0
    param.alpha_T = 1.0
    param.alpha_X = 0.0

    # mesh generation
    mesh = sfem.rectangle_uniform_trimesh(3.0, 1.0, NX+1, NY+1)

    # create OCP class
    ocp = nse.OCP(param=param, mesh=mesh, femtype=femtype)
    ocp.null_control = get_null_control_nodes(mesh, XLIM, YLIM)

    return ocp


def get_random_force(size, zero_indices):
    """
    Returns a random initial force with values on [-1, 1].
    """
    init_u = 2 * np.random.rand(size) - 1
    init_u[zero_indices] = 0.0

    return init_u


def state_solve(ocp):
    """
    Returns solution of NSE without delay. See also <nsedelay.State_Solver>.
    """
    return nse.State_Solver(ocp.data.init_u, ocp.data.hist_u,
        ocp.init_control, ocp.param, ocp.mesh, ocp.tgrid, ocp.femstruct,
        ocp.noslipbc, ocp.Asolve, ocp.M, info=False)


def time_evol(ocp, var):
    """
    Returns the time series for the L2-norm of the data var at each time node.
    """
    N = ocp.mesh.dof
    var_norm = np.zeros(var.shape[1]).astype(float)
    for t in range(len(var_norm)):
        var_norm[t] \
            = np.sqrt(np.dot(var[:N, t], ocp.M * var[:N, t])
            + np.dot(var[N:, t], ocp.M * var[N:, t]))
    return var_norm


def get_init_data(ocp):
    """
    Returns the initial data. Calculated by solving a steady Stokes flow with
    random force.
    """
    from scipy import sparse as sp
    from scipy.sparse.linalg import spsolve

    # matrix assembly
    K, M, Bx, By, _ = sfem.assemble(ocp.mesh, ocp.femstruct)

    # penalty matrix
    Z = sp.spdiags(1e-10*np.ones(ocp.mesh.num_node), 0, ocp.mesh.num_node,
        ocp.mesh.num_node)

    # total matrix
    Atotal = sp.bmat([[ocp.param.nu * K, None, -Bx.T],
                     [None, ocp.param.nu * K, -By.T],
                     [Bx, By, Z]], format='csr')

    # Apply no-slip boundary condition to total matrix
    Atotal = sfem.apply_noslip_bc(Atotal, ocp.noslipbc)

    try:
        init_data = 10 * np.load(os.getcwd() + '/init_data.npy')
        print("\nInitial data loaded succesfully.")
    except FileNotFoundError:
        init_data = get_random_force(2*ocp.mesh.dof, ocp.noslipbc)
        print("\nRandom initial data generated.\n")
        np.save('init_data.npy', init_data)

    rhs = np.hstack([M * init_data[:ocp.mesh.dof], M * init_data[ocp.mesh.dof:],
        np.zeros(ocp.mesh.num_node)])
    init_data = spsolve(Atotal, rhs)[:2*ocp.mesh.dof]
    init_data_max = max([abs(init_data.max()), abs(init_data.min())])
    print("Maximum norm of initial data: {:.12f}\n".format(init_data_max))

    return init_data


def main(femtype):
    """
    Main function.
    """
    # check or create directory
    check_dir()

    # Bounding box for the control region.
    XLIM = [0.5, 2.5]
    YLIM = [0.25, 0.75]

    # Number of nodes (plus 1) in X and Y axis.
    NX, NY = 121, 41

    # Load or generate random initial data.
    ocp = get_ocp(NX, NY, 0.0, XLIM, YLIM, femtype)
    init_data = get_init_data(ocp)
    ocp.data.init_u = init_data

    # solution of NSE without delay
    undelayed_state = state_solve(ocp)

    # OCP class with delay
    ocp = get_ocp(NX, NY, 0.5, XLIM, YLIM, femtype)
    ocp.print()
    ocp.data.init_u = init_data
    ocp.data.goal_u = undelayed_state[:, 1:]
    hist_shape = ocp.data.hist_u.shape
    ocp.data.hist_u = 0.5 * ocp.data.hist_u

    # solution of NSE with delay
    delayed_state = state_solve(ocp)

    # solution to the optimal control problem
    optimal_state, optimal_adjoint, optimal_control, residual, COST_VALUES, \
        OPT_REST_NORM, REL_ERROR = nse.barzilai_borwein(ocp, version=3)

    # residual error
    time_evol_residual_error = time_evol(ocp, residual)

    # local residual error
    time_evol_residual_error_local = residual.copy()
    time_evol_residual_error_local[ocp.null_control, :] = 0
    time_evol_residual_error_local = time_evol(ocp, time_evol_residual_error_local)

    # optimality residual
    time_evol_optimres = ocp.param.alpha * optimal_control + optimal_adjoint[:, :-1]
    time_evol_optimres[ocp.null_control, :] = 0.0
    time_evol_optimres_norm = time_evol(ocp, time_evol_optimres)

    # control norm
    time_evol_control_norm = time_evol(ocp, optimal_control)

    # save npy files
    FILEPATH = os.getcwd() + '/npyfiles_velocity/'
    np.save(FILEPATH + 'init_hist.npy', ocp.data.hist_u)
    np.save(FILEPATH + 'goal_state.npy', ocp.data.goal_u)
    np.save(FILEPATH + 'delay_state.npy', delayed_state[:, 1:])
    np.save(FILEPATH + 'optim_state.npy', optimal_state[:, 1:])
    np.save(FILEPATH + 'optim_adjnt.npy', optimal_adjoint[:, :-1])
    np.save(FILEPATH + 'optim_cntrl.npy', optimal_control)
    np.save(FILEPATH + 'time_evol_residual_error.npy',
        time_evol_residual_error)
    np.save(FILEPATH + 'time_evol_residual_error_local.npy',
        time_evol_residual_error_local)
    np.save(FILEPATH + 'time_evol_optimres_norm.npy',
        time_evol_optimres_norm)
    np.save(FILEPATH + 'time_evol_control_norm.npy',
        time_evol_control_norm)
    np.save(FILEPATH + 'bb_cost_values.npy', np.array(COST_VALUES).astype(float))
    np.save(FILEPATH + 'bb_opt_res_norm.npy', np.array(OPT_REST_NORM).astype(float))
    np.save(FILEPATH + 'bb_rel_error.npy', np.array(REL_ERROR).astype(float))

    dict_data = {'MESH_NUM_NODE': ocp.mesh.num_node, 'MESH_DOF': ocp.mesh.dof,
        'TIME_MESH': ocp.tgrid.mesh, 'XLIM': XLIM, 'YLIM': YLIM, 'NX': NX, 'NY': NY}

    np.save(FILEPATH + 'dict_data.npy', dict_data)


if __name__ == '__main__':
    utils.print_start_implementation()
    main('taylor_hood')
    utils.print_end_implementation()

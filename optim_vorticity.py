# -*- coding: utf-8 -*-
"""
This python script implements the vorticity minimization problem for the optimal
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
from optim_velocity import get_init_data
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
    Check if there is a directory 'npyfiles_vorticity', otherwise, create one.
    """
    if os.path.isdir('npyfiles_vorticity'):
        pass
    else:
        os.mkdir('npyfiles_vorticity')


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
    param.ocptol = 1e-4
    param.alpha = 1e-3
    param.alpha_O = 0.0
    param.alpha_T = 0.0
    param.alpha_X = 0.1

    # mesh generation
    mesh = sfem.rectangle_uniform_trimesh(3.0, 1.0, NX+1, NY+1)

    # create OCP class
    ocp = nse.OCP(param=param, mesh=mesh, femtype=femtype)
    ocp.null_control = get_null_control_nodes(mesh, XLIM, YLIM)

    return ocp


def state_solve(ocp):
    """
    Returns solution of NSE without delay. See also <nsedelay.State_Solver>.
    """
    return nse.State_Solver(ocp.data.init_u, ocp.data.hist_u,
        ocp.init_control, ocp.param, ocp.mesh, ocp.tgrid, ocp.femstruct,
        ocp.noslipbc, ocp.Asolve, ocp.M, info=False)


def time_evol(ocp, var):
    """
    Returns the time series data.
    """
    N = ocp.mesh.dof
    var_norm = np.zeros(var.shape[1]).astype(float)
    for t in range(len(var_norm)):
        var_norm[t] \
            = np.sqrt(np.dot(var[:N, t], ocp.M * var[:N, t])
            + np.dot(var[N:, t], ocp.M * var[N:, t]))
    return var_norm


def time_evol_vorticity(ocp, var):
    """
    Returns the time series for the L2-norm of the data var at each time node.
    """
    N = ocp.mesh.dof
    var_norm = np.zeros(var.shape[1]).astype(float)
    for t in range(len(var_norm)):
        var_norm[t] \
            = np.sqrt(np.dot(var[:, t], ocp.V * var[:, t]))
    return var_norm


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

    # OCP class with delay
    ocp = get_ocp(NX, NY, 0.5, XLIM, YLIM, femtype)
    ocp.print()
    ocp.data.hist_u = 0.5 * ocp.data.hist_u

    # Load or generate random initial data.
    ocp.data.init_u = get_init_data(ocp)

    # optimal control problem
    optimal_state, optimal_adjoint, optimal_control, residual, COST_VALUES, \
        OPT_REST_NORM, REL_ERROR = nse.barzilai_borwein(ocp, version=3)

    # optimality residual
    time_evol_optimres = ocp.param.alpha * optimal_control + optimal_adjoint[:, :-1]
    time_evol_optimres[ocp.null_control, :] = 0.0
    time_evol_optimres_norm = time_evol(ocp, time_evol_optimres)

    # control norm
    time_evol_control_norm = time_evol(ocp, optimal_control)

    # vorticity norm
    time_evol_vorticity_norm = time_evol_vorticity(ocp, optimal_state[:, 1:])

    # local vorticity norm
    time_evol_vorticity_local = optimal_state.copy()[:, 1:]
    time_evol_vorticity_local[ocp.null_control, :] = 0
    time_evol_vorticity_local = time_evol_vorticity(ocp, time_evol_vorticity_local)

    # save npy files
    FILEPATH = os.getcwd() + '/npyfiles_vorticity/'
    np.save(FILEPATH + 'init_hist.npy', ocp.data.hist_u)
    np.save(FILEPATH + 'optim_state.npy', optimal_state[:, 1:])
    np.save(FILEPATH + 'optim_adjnt.npy', optimal_adjoint[:, :-1])
    np.save(FILEPATH + 'optim_cntrl.npy', optimal_control)
    np.save(FILEPATH + 'time_evol_optimres_norm.npy',
        time_evol_optimres_norm)
    np.save(FILEPATH + 'time_evol_control_norm.npy',
        time_evol_control_norm)
    np.save(FILEPATH + 'time_evol_vorticity_norm',
        time_evol_vorticity_norm)
    np.save(FILEPATH + 'time_evol_vorticity_local',
        time_evol_vorticity_local)
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

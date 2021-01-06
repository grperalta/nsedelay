# -*- coding: utf-8 -*-
"""
This python script runs tests for the order of convergences for the optimal
control of the delayed Navier-Stokes equation using the implicit-explicit
(IMEX) Euler scheme.

For more details, refer to the manuscript:
    'Optimal Control for the Navier-Stokes Equation with Time Delay in the
    Convection: Analysis and Finite Element Approximations' by Gilbert Peralta
    and John Sebastian Simon.

Gilbert Peralta
Department of Mathematics and Computer Science
University of the Philippines Baguio
Governor Pack Road, Baguio, Philippines 2600
Email: grperalta@up.edu.ph
7 January 2021
"""

from __future__ import division
import stokesfem as sfem
import functions as fun
import nsedelay as nse
import numpy as np
import utils

__author__ = "Gilbert Peralta"
__institution__ = "University of the Philippines Baguio"
__email__ = "grperalta@up.edu.ph"
__date__ = "7 January 2021"


def print_order_convergence(list_error, list_alpha):
    """
    Table for the mesh sizes, L2-error norms and order of convergences.
    """
    print(utils.Colors.UNDERLINE + ' '*80 + utils.Colors.ENDC)
    print("\t\t\t{}\n".format(list_error[0]))
    print("Tikhonov Parameter\tError\t\t\tOrder of Convergence")
    print(utils.Colors.UNDERLINE + ' '*80 + utils.Colors.ENDC)

    list_error = list_error[1:]
    for k in range(len(list_alpha)):
        if k == 0:
            print("{:.10e}\t{:.10e}".format(
                list_alpha[k], list_error[k]))
        else:
            eoc = np.log(list_error[k-1] / list_error[k]) \
                / np.log(list_alpha[k-1] / list_alpha[k])
            print("{:.10e}\t{:.10e}\t{:.10e}".format(
                list_alpha[k], list_error[k], eoc))

    print(utils.Colors.UNDERLINE + ' '*80 + utils.Colors.ENDC)


def get_exact_data(ocp, femtype='p1_bubble'):
    """
    Returns the exact optimal state, numerical optimal adjoint, numerical
    optimal control and forcing function.
    """

    exact_state = np.zeros((2*ocp.mesh.dof, ocp.param.Nt)).astype(float)
    exact_force = np.zeros((2*ocp.mesh.dof, ocp.param.Nt)).astype(float)

    if femtype is 'p1_bubble':
        exact_state[:ocp.mesh.dof, :] \
            = fun.U_EXACT_BUBBLE(ocp.tgrid.mesh, ocp.mesh)
        exact_state[ocp.mesh.dof:, :] \
            = fun.V_EXACT_BUBBLE(ocp.tgrid.mesh, ocp.mesh)
        exact_force[:ocp.mesh.dof, :] \
            = fun.F_EXACT_BUBBLE_DELAY(ocp.tgrid.mesh, ocp.param.tau, ocp.mesh)
        exact_force[ocp.mesh.dof:, :] \
            = fun.G_EXACT_BUBBLE_DELAY(ocp.tgrid.mesh, ocp.param.tau, ocp.mesh)
    else:
        mesh_all_node = ocp.mesh.all_node()
        exact_state[:ocp.mesh.dof, :] \
            = fun.U_EXACT(ocp.tgrid.mesh, mesh_all_node[:, 0], mesh_all_node[:,1])
        exact_state[ocp.mesh.dof:, :] \
            = fun.V_EXACT(ocp.tgrid.mesh, mesh_all_node[:, 0], mesh_all_node[:,1])
        exact_force[:ocp.mesh.dof, :] \
            = fun.F_EXACT_DELAY(ocp.tgrid.mesh, ocp.param.tau,
            mesh_all_node[:, 0], mesh_all_node[:,1])
        exact_force[ocp.mesh.dof:, :] \
            = fun.G_EXACT_DELAY(ocp.tgrid.mesh, ocp.param.tau,
            mesh_all_node[:, 0], mesh_all_node[:,1])

    exact_adjoint \
        = nse.Adjoint_Solver(2*exact_state[:, 1:], ocp.data.hist_u,
        exact_state, ocp.param, ocp.mesh, ocp.tgrid, ocp.femstruct,
        ocp.noslipbc, ocp.Asolve, ocp.M, ocp.V, info=False)

    exact_control = - (1.0/ocp.param.alpha) * exact_adjoint[:, :-1]
    exact_force = exact_force[:, 1:] - exact_control

    return exact_state, exact_adjoint, exact_control, exact_force


def get_ocp(NX=81, NT=641, alpha=1e-1, ocptol=1e-6, femtype='p1_bubble'):
    """
    Returns the nsedelay.OCP class for the optimal control problem. Here,
    NX is the number of nodes on a side of a unit square and NT is the number
    of time nodes.
    """

    # parameters in OCP
    param = nse.Parameters.get_default()
    param.Nt = NT
    param.alpha = alpha
    param.ocptol = ocptol

    # OCP class
    ocp = nse.OCP(param=param, mesh=sfem.square_uniform_trimesh(NX),
        femtype=femtype)
    exact_state, exact_adjoint, exact_control, ocp.force \
        = get_exact_data(ocp, femtype=femtype)
    ocp.data.goal_u = - exact_state[:, 1:]

    return exact_state, exact_adjoint, exact_control, ocp


def main(femtype='p1_bubble'):
    """
    Main function.
    """
    utils.print_start_implementation()
    LIST_ALPHA = [1.0, 1e-1, 1e-2, 1e-3]

    LIST_L2_ERROR_STATE = ['L2-ERROR ON OPTIMAL STATE']
    LIST_L2_ERROR_ADJOINT = ['L2-ERROR ON OPTIMAL ADJOINT']
    LIST_L2_ERROR_CONTROL = ['L2-ERROR ON OPTIMAL CONTROL']

    LIST_L2H1_ERROR_STATE = ['L2(H1)-ERROR ON OPTIMAL STATE']
    LIST_L2H1_ERROR_ADJOINT = ['L2(H1)-ERROR ON OPTIMAL ADJOINT']
    LIST_L2H1_ERROR_CONTROL = ['L2(H1)-ERROR ON OPTIMAL CONTROL']

    LIST_LINFL2_ERROR_STATE = ['LINF(L2)-ERROR ON OPTIMAL STATE']
    LIST_LINFL2_ERROR_ADJOINT = ['LINF(L2)-ERROR ON OPTIMAL ADJOINT']
    LIST_LINFL2_ERROR_CONTROL = ['LINF(L2)-ERROR ON OPTIMAL CONTROL']

    for k in range(len(LIST_ALPHA)):
        # exact data and OCP classs
        exact_state, exact_adjoint, exact_control, ocp \
            = get_ocp(alpha=LIST_ALPHA[k], femtype=femtype)
        ocp.print()

        # numerical solution
        state, adjoint, control, _, _, _, _ \
            = nse.barzilai_borwein(ocp)

        LIST_L2_ERROR_STATE \
            += [nse.L2_norm(state - exact_state, ocp.param, ocp.mesh, ocp.M)]
        LIST_L2_ERROR_ADJOINT \
            += [nse.L2_norm(adjoint - exact_adjoint, ocp.param, ocp.mesh, ocp.M)]
        LIST_L2_ERROR_CONTROL \
            += [nse.L2_norm(control - exact_control, ocp.param, ocp.mesh, ocp.M)]

        LIST_L2H1_ERROR_STATE \
            += [nse.L2H1_norm(state - exact_state, ocp.param, ocp.mesh, ocp.K)]
        LIST_L2H1_ERROR_ADJOINT \
            += [nse.L2H1_norm(adjoint - exact_adjoint, ocp.param, ocp.mesh, ocp.K)]
        LIST_L2H1_ERROR_CONTROL \
            += [nse.L2H1_norm(control - exact_control, ocp.param, ocp.mesh, ocp.K)]

        LIST_LINFL2_ERROR_STATE \
            += [nse.LinfL2_norm(state - exact_state, ocp.param, ocp.mesh, ocp.M)]
        LIST_LINFL2_ERROR_ADJOINT \
            += [nse.LinfL2_norm(adjoint - exact_adjoint, ocp.param, ocp.mesh, ocp.M)]
        LIST_LINFL2_ERROR_CONTROL \
            += [nse.LinfL2_norm(control - exact_control, ocp.param, ocp.mesh, ocp.M)]

    for LIST in [LIST_L2_ERROR_STATE, LIST_L2_ERROR_ADJOINT,
        LIST_L2_ERROR_CONTROL]:
        print_order_convergence(LIST, LIST_ALPHA)

    for LIST in [LIST_L2H1_ERROR_STATE, LIST_L2H1_ERROR_ADJOINT,
        LIST_L2H1_ERROR_CONTROL]:
        print_order_convergence(LIST, LIST_ALPHA)

    for LIST in [LIST_LINFL2_ERROR_STATE, LIST_LINFL2_ERROR_ADJOINT,
        LIST_LINFL2_ERROR_CONTROL]:
        print_order_convergence(LIST, LIST_ALPHA)

    utils.print_end_implementation()


if __name__ == '__main__':
    # main('p1_bubble')
    main('taylor_hood')

# -*- coding: utf-8 -*-
"""
This python script runs test for the implicit-explicit (IMEX) Euler scheme
for the Navier-Stokes equation with delay in the convection term using the
Taylor-Hood and Mini-Finite elements.

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
from demo_nse import get_nse_fem_matrices
from demo_nse import L2_spacetime_error
import stokesfem as sfem
import functions as fun
import numpy as np
import utils

__author__ = "Gilbert Peralta"
__institution__ = "University of the Philippines Baguio"
__email__ = "grperalta@up.edu.ph"
__date__ = "9 July 2020"


def navier_stokes(name='taylor_hood'):
    """
    Test problem for the IMEX Euler scheme for NSE.

    Parameter
    ---------
    name : str
        Either 'taylor_hood' or 'p1_bubble'.
    """
    print("\nIMEX EULER SCHEME FOR DELAYED NAVIER-STOKES EQUATION "
        + "VIA {} FEM".format(name.upper()))
    list_Nx, list_Nt, velocity_error, pressure_error, mesh_sizes \
        = [9, 17, 33], [11, 41, 161], [], [], []

    for i in range(len(list_Nx)):
        # temporal mesh
        T, r = 1.0, 0.5
        t = np.linspace(0, 1, list_Nt[i])
        dt = t[1] - t[0]
        num_hist = int(r / dt)

        # history temporal coefficient
        histgrid = 0.5 * (fun.c(t[:num_hist] - r) + fun.c(t[1:num_hist+1] - r))
        # mesh generation
        mesh = sfem.square_uniform_trimesh(list_Nx[i]).femprocess(name=name)

        # exact data
        if name is 'taylor_hood':
            u_exact = fun.U_EXACT(t, mesh.all_node()[:, 0], mesh.all_node()[:, 1])
            v_exact = fun.V_EXACT(t, mesh.all_node()[:, 0], mesh.all_node()[:, 1])
            force_x = fun.F_EXACT_DELAY(t, r,
                mesh.all_node()[:, 0], mesh.all_node()[:, 1])
            force_y = fun.G_EXACT_DELAY(t, r,
                mesh.all_node()[:, 0], mesh.all_node()[:, 1])
            u_histo = \
                fun.U_HIST(histgrid, mesh.all_node()[:, 0], mesh.all_node()[:, 1])
            v_histo = \
                fun.V_HIST(histgrid, mesh.all_node()[:, 0], mesh.all_node()[:, 1])
        else:
            u_exact = fun.U_EXACT_BUBBLE(t, mesh)
            v_exact = fun.V_EXACT_BUBBLE(t, mesh)
            force_x = fun.F_EXACT_BUBBLE_DELAY(t, r, mesh)
            force_y = fun.G_EXACT_BUBBLE_DELAY(t, r, mesh)
            u_histo = fun.U_HIST_BUBBLE(histgrid, mesh)
            v_histo = fun.U_HIST_BUBBLE(histgrid, mesh)
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
            if it < num_hist:
                N = sfem.convection(mesh, u_histo[:, it], v_histo[:, it],
                    femdatastruct)
            else:
                N = sfem.convection(mesh, u_app[:, it - num_hist],
                    v_app[:, it - num_hist], femdatastruct)

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

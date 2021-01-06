# -*- coding: utf-8 -*-
"""
This python script plots the results of the optim_vorticity.py script.

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
from matplotlib import cm
from matplotlib import rc
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import nsedelay as nse
import stokesfem as sfem
import os

__author__ = "Gilbert Peralta"
__institution__ = "University of the Philippines Baguio"
__email__ = "grperalta@up.edu.ph"
__date__ = "7 January 2021"


FILEPATH = os.getcwd() + '/npyfiles_vorticity/'
DATA = np.load(FILEPATH + 'dict_data.npy', encoding='latin1', allow_pickle=True)[()]

rc('font',**{'family':'DejaVu Sans', 'serif':['Computer Modern Roman']})
rc('text', usetex=True)


def bounding_box(ax):
    """
    Plot bounding box of control region.
    """
    ax.plot([DATA['NX']*(DATA['XLIM'][0]/3), DATA['NX']*(DATA['XLIM'][1]/3)],
        [DATA['NY']*DATA['YLIM'][0], DATA['NY']*DATA['YLIM'][0]],
        color='k', lw=0.5)
    ax.plot([DATA['NX']*(DATA['XLIM'][0]/3), DATA['NX']*(DATA['XLIM'][1]/3)],
        [DATA['NY']*DATA['YLIM'][1], DATA['NY']*DATA['YLIM'][1]],
        color='k', lw=0.5)
    ax.plot([DATA['NX']*(DATA['XLIM'][0]/3), DATA['NX']*(DATA['XLIM'][0]/3)],
        [DATA['NY']*DATA['YLIM'][0], DATA['NY']*DATA['YLIM'][1]],
        color='k', lw=0.5)
    ax.plot([DATA['NX']*(DATA['XLIM'][1]/3), DATA['NX']*(DATA['XLIM'][1]/3)],
        [DATA['NY']*DATA['YLIM'][0], DATA['NY']*DATA['YLIM'][1]],
        color='k', lw=0.5)
    return None


def sample_plots(data, fig_num=1):
    """
    Displays some plots on the Euclidean norm of the input data.
    """
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 2.1))
    k = 0
    TITLE = ['(a)', '(b)']
    for ax in axes.flat:
        bounding_box(ax)
        if k == 1:
            image = ax.imshow(data[k].reshape(DATA['NX']+1,DATA['NY']+1).T,
                interpolation='quadric', cmap=cm.RdBu_r,
                origin='lower', vmin=data[k].min(), vmax=data[k].max())
            ax.contour(data[k].reshape(DATA['NX']+1,DATA['NY']+1).T, cmap=cm.RdBu_r,
                vmin=data[k].min(), vmax=data[k].max(), levels=100)
        else:
            image = ax.imshow(data[k].reshape(DATA['NX']+1,DATA['NY']+1).T,
                interpolation='quadric', cmap=cm.Blues_r,
                origin='lower', vmin=data[k].min(), vmax=0.2)
        ax.set_xlabel(TITLE[k], fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(image, ax=ax, orientation='vertical',
            fraction=0.0165, pad=0.02)
        k += 1
    plt.tight_layout()


def get_data(data, time_node):
    """
    Return data for imshow.
    """
    image_data = np.sqrt(data[:DATA['MESH_NUM_NODE'], time_node]**2 \
            + data[DATA['MESH_DOF']:DATA['MESH_DOF']+DATA['MESH_NUM_NODE'], time_node]**2)
    return image_data


def plot_residual_error():
    """
    Plot results for the BB method.
    """
    fig = plt.figure(figsize=(9,5))

    ax1 = fig.add_subplot(221)
    time_evol_vorticity_norm = np.load(FILEPATH + 'time_evol_vorticity_norm.npy')
    time_evol_vorticity_local = np.load(FILEPATH + 'time_evol_vorticity_local.npy')
    ax1.semilogy(DATA['TIME_MESH'][1: ], time_evol_vorticity_norm, color='b',
        label=r'$\|\nabla \times u_\sigma^\star(t)\|_{L^2(\Omega)}$', lw=1.5)
    ax1.semilogy(DATA['TIME_MESH'][1: ], time_evol_vorticity_local, color='r',
        label=r'$\|\nabla \times u_\sigma^\star(t)\|_{L^2(\omega)}$',
        ls='dashdot', lw=1.5)
    ax1.autoscale(enable=True, axis='x', tight=True)
    plt.grid(True, which="major", ls="dotted", color='0.65')
    ax1.legend(loc='best', fontsize='12')
    ax1.set_xlabel('(a)', fontsize=12)
    ax1.set_yticks([0.1, 1, 10])

    ax2 = fig.add_subplot(222)
    cost_values = np.load(FILEPATH + 'bb_cost_values.npy')
    ax2.semilogy(range(1, len(cost_values)+1), cost_values, color='b',
        label=r'$j_{\sigma, \varepsilon_p}(q_\sigma^{(\ell)})$', lw=1.5)
    opt_res_norm = np.load(FILEPATH + 'bb_opt_res_norm.npy')
    ax2.autoscale(enable=True, axis='x', tight=True)
    plt.grid(True, which="both", ls="dotted", color='0.65')
    ax2.legend(loc='best', fontsize='12')
    ax2.set_xlabel('(b)', fontsize=12)

    ax3 = fig.add_subplot(223)
    time_evol_control_norm = np.load(FILEPATH + 'time_evol_control_norm.npy')
    ax3.semilogy(DATA['TIME_MESH'][1: ], time_evol_control_norm, color='b',
        label=r'$\|q_\sigma^\star(t)\|_{L^2(\omega)^2}$', lw=1.5)
    ax3.autoscale(enable=True, axis='x', tight=True)
    plt.grid(True, which="major", ls="dotted", color='0.65')
    ax3.legend(loc='best', fontsize='12')
    ax3.set_xlabel('(c)', fontsize=12)
    ax3.set_yticks([0.01, 0.1, 1, 10])

    ax4 = fig.add_subplot(224)
    opt_res_norm = np.load(FILEPATH + 'bb_opt_res_norm.npy')
    ax4.semilogy(range(1, len(cost_values)+1), opt_res_norm, color='b',
        label=r'$\|\alpha q_\sigma^{(\ell)} + w_\sigma^{(\ell)}\|_{L^2(I \times \omega)^2}$',
        lw=1.5)
    opt_res_norm = np.load(FILEPATH + 'bb_opt_res_norm.npy')
    ax4.autoscale(enable=True, axis='x', tight=True)
    plt.grid(True, which="major", ls="dotted", color='0.65')
    ax4.legend(loc='best', fontsize='12')
    ax4.set_xlabel('(d)', fontsize=12)

    plt.tight_layout()


def plot_optimal_solution(time_node):
    """
    Plot of optimal control and optimal state.
    """
    optimal_state = get_data(np.load(FILEPATH + 'optim_state.npy'), time_node)
    optimal_control = get_data(np.load(FILEPATH + 'optim_cntrl.npy'), time_node)

    sample_plots([optimal_state, optimal_control])


if __name__ == '__main__':
    plot_optimal_solution(9)
    plot_optimal_solution(49)
    plot_optimal_solution(99)
    plot_residual_error()
    plt.show()

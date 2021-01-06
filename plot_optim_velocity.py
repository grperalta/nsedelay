# -*- coding: utf-8 -*-
"""
This python script plots the results of the optim_velocity.py script.

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
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import nsedelay as nse
import stokesfem as sfem
import os

__author__ = "Gilbert Peralta"
__institution__ = "University of the Philippines Baguio"
__email__ = "grperalta@up.edu.ph"
__date__ = "7 January 2021"


FILEPATH = os.getcwd() + '/npyfiles_velocity/'
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


def sample_plots(data, image_vmin, image_vmax, fig_num=1):
    """
    Displays some plots on the Euclidean norm of the input data.
    """
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 4.2))
    k = 0
    TITLE = ['(a)', '(b)', '(c)', '(d)']
    cmap = cm.Blues_r
    for ax in axes.flat:
        bounding_box(ax)
        if k == 3:
            image_vmin = data[k].min()
            image_vmax = data[k].max()
            cmap = cm.RdBu_r
            ax.contour(data[k].reshape(DATA['NX']+1,DATA['NY']+1).T, cmap=cm.RdBu_r,
                vmin=data[k].min(), vmax=data[k].max(), levels=100)
        image = ax.imshow(data[k].reshape(DATA['NX']+1,DATA['NY']+1).T,
            interpolation='quadric', cmap=cmap,
            origin='lower', vmin=image_vmin, vmax=image_vmax)
        ax.set_xlabel(TITLE[k], fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(image, ax=ax, orientation='vertical', fraction=0.0165, pad=0.02)
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
    time_evol_residual_error = np.load(FILEPATH + 'time_evol_residual_error.npy')
    time_evol_residual_error_local \
        = np.load(FILEPATH + 'time_evol_residual_error_local.npy')
    ax1.semilogy(DATA['TIME_MESH'][1: ], time_evol_residual_error, color='b',
        label=r'$\|u_\sigma^\star(t) - u_{d\sigma}(t)\|_{L^2(\Omega)^2}$', lw=1.5)
    ax1.semilogy(DATA['TIME_MESH'][1: ], time_evol_residual_error_local, color='r',
        label=r'$\|u_\sigma^\star(t) - u_{d\sigma}(t)\|_{L^2(\omega)^2}$',
        ls='dashdot', lw=1.5)
    ax1.autoscale(enable=True, axis='x', tight=True)
    plt.grid(True, which="major", ls="dotted", color='0.65')
    ax1.legend(loc='best', fontsize='12')
    ax1.set_xlabel('(a)', fontsize=12)

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
    ax3.set_yticks([0.1, 1, 10])

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


def plot_optimal_solution(time_nodes):
    """
    Plot of optimal control, optimal state, desired state and uncontrolled state.
    """
    undelayed_state = []
    delayed_state = []
    optimal_state = []
    optimal_control = []
    list_image_vmin = []
    list_image_vmax = []
    for k in range(len(time_nodes)):
        undelayed_state += [get_data(np.load(FILEPATH + 'goal_state.npy'),
            time_nodes[k])]
        delayed_state += [get_data(np.load(FILEPATH + 'delay_state.npy'),
            time_nodes[k])]
        optimal_state += [get_data(np.load(FILEPATH + 'optim_state.npy'),
            time_nodes[k])]
        optimal_control += [get_data(np.load(FILEPATH + 'optim_cntrl.npy'),
            time_nodes[k])]
        list_image_vmin += [min([undelayed_state[k].min(), delayed_state[k].min(),
            optimal_state[k].min()])]
        list_image_vmax += [max([undelayed_state[k].max(), delayed_state[k].max(),
            optimal_state[k].max()])]
    image_vmin = min(list_image_vmin)
    image_vmax = max(list_image_vmax)
    for k in range(len(time_nodes)):
        sample_plots([undelayed_state[k], delayed_state[k], optimal_state[k],
            optimal_control[k]], image_vmin, image_vmax)


if __name__ == '__main__':
    plot_optimal_solution([9, 49, 99])
    plot_residual_error()
    plt.show()

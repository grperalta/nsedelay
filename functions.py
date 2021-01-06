# -*- coding: utf-8 -*-
"""
The functions and their derivatives used in the tests.
"""

from stokesfem import bubble_interpolation
import numpy as np

__author__ = "Gilbert Peralta"
__institution__ = "University of the Philippines Baguio"
__email__ = "grperalta@up.edu.ph"
__date__ = "9 July 2020"

PI2 = 2.0 * np.math.pi
VISCO = 1.0

def u(x, y):
    """
    X-component of velocity.
    """
    return (1.0 - np.cos(PI2*x)) * np.sin(PI2*y)


def v(x,y):
    """
    Y-component of velocity.
    """
    return np.sin(PI2*x) * (np.cos(PI2*y) - 1.0)


def u_x(x, y):
    """
    Derivative wrt to x of the x-component of velocity.
    """
    return PI2 * np.sin(PI2*x) * np.sin(PI2*y)


def u_y(x, y):
    """
    Derivative wrt to y of the x-component of velocity.
    """
    return PI2 * (1.0 - np.cos(PI2*x)) * np.cos(PI2*y)


def u_xy(x, y):
    """
    Derivative wrt to xy of the x-component of velocity.
    """
    return PI2 * PI2 * np.sin(PI2*x) * np.cos(PI2*y)


def u_yy(x, y):
    """
    Derivative wrt to yy of the x-component of velocity.
    """
    return - PI2 * PI2 * (1.0 - np.cos(PI2*x)) * np.sin(PI2*y)


def v_x(x, y):
    """
    Derivative wrt to x of the y-component of velocity.
    """
    return PI2 * np.cos(PI2*x) * (np.cos(PI2*y) - 1.0)


def v_y(x, y):
    """
    Derivative wrt to y of the y-component of velocity.
    """
    return - PI2 * np.sin(PI2*x) * np.sin(PI2*y)


def v_xx(x, y):
    """
    Derivative wrt to x of the y-component of velocity.
    """
    return - PI2 * PI2 * np.sin(PI2*x) * (np.cos(PI2*y) - 1.0)


def v_xy(x, y):
    """
    Derivative wrt to yx of the y-component of velocity.
    """
    return - PI2 * PI2 * np.cos(PI2*x) * np.sin(PI2*y)


def delta_u(x, y):
    """
    Laplacian of the x-component of velocity.
    """
    return PI2 * PI2 * np.sin(PI2*y) * (2.0 * np.cos(PI2*x) - 1.0)


def delta_v(x, y):
    """
    Laplacian of the y-component of velocity.
    """
    return - PI2 * PI2 * np.sin(PI2*x) * (2.0 * np.cos(PI2*y) -1.0)


def U_curl(x, y):
    """
    Curl of the velocity field U = (u, v).
    """
    return u_y(x, y) - v_x(x, y)


def U_curlcurl_x(x, y):
    """
    X-component of Curl-Curl of the velocity field U = (u, v).
    """
    return v_xy(x, y) - u_yy(x, y)


def U_curlcurl_y(x, y):
    """
    Y-component of Curl-Curl of the velocity field U = (u, v).
    """
    return u_xy(x, y) - v_xx(x, y)


def p(x, y):
    """
    Pressure.
    """
    return np.cos(PI2*y) - np.cos(PI2*x)


def p_x(x, y):
    """
    Derivative wrt x of pressure.
    """
    return PI2 * np.sin(PI2*x)


def p_y(x, y):
    """
    Derivative wrt y of pressure.
    """
    return - PI2 * np.sin(PI2*y)


def f(x, y):
    """
    The function - Delta u + p_x.
    """
    return - delta_u(x, y) + p_x(x, y)


def g(x, y):
    """
    The function - Delta v + p_y.
    """
    return - delta_v(x, y) + p_y(x, y)


def c(t):
    """
    Temporal velocity coefficient.
    """
    return np.cos(np.math.pi*t)


def cp(t):
    """
    Temporal pressure coefficient.
    """
    return np.sin(np.math.pi*t)


def dc(t):
    """
    Derivative of temporal velocity coefficient.
    """
    return - np.math.pi * np.sin(np.math.pi*t)


def U_EXACT(t, x, y):
    """
    Exact X-component of velocity.
    """
    return np.outer(u(x,y), c(t))


def U_EXACT_BUBBLE(t, mesh):
    """
    Exact X-component of velocity with P1-Bubble interpolation.
    """
    return np.outer(bubble_interpolation(mesh, u), c(t))


def V_EXACT(t, x, y):
    """
    Exact Y-component of velocity.
    """
    return np.outer(v(x, y), c(t))


def V_EXACT_BUBBLE(t, mesh):
    """
    Exact Y-component of velocity with P1-Bubble interpolation.
    """
    return np.outer(bubble_interpolation(mesh, v), c(t))


def P_EXACT(t, x, y):
    """
    Exact presure.
    """
    return np.outer(p(x, y), cp(t))


def F_EXACT(t, x, y):
    """
    Exact X-component of forcing function.
    """
    UT = np.outer(u(x, y), dc(t))
    D2U = np.outer(delta_u(x, y), c(t))
    PX = np.outer(p_x(x, y), cp(t))
    UDU = np.outer(u(x, y), c(t)) * np.outer(u_x(x, y), c(t)) \
        + np.outer(v(x, y), c(t)) * np.outer(u_y(x, y), c(t))
    return UT - VISCO * D2U + UDU + PX


def F_EXACT_BUBBLE(t, mesh):
    """
    Exact X-component of forcing function with P1-Bubble interpolation.
    """
    UT = np.outer(bubble_interpolation(mesh, u), dc(t))
    D2U = np.outer(bubble_interpolation(mesh, delta_u), c(t))
    PX = np.outer(bubble_interpolation(mesh, p_x), cp(t))
    UDU = np.outer(bubble_interpolation(mesh, u), c(t)) \
        * np.outer(bubble_interpolation(mesh, u_x), c(t)) \
        + np.outer(bubble_interpolation(mesh, v), c(t)) \
        * np.outer(bubble_interpolation(mesh, u_y), c(t))
    return UT - VISCO * D2U + UDU + PX


def G_EXACT(t, x, y):
    """
    Exact Y-component of forcing function.
    """
    VT = np.outer(v(x, y), dc(t))
    D2V = np.outer(delta_v(x, y), c(t))
    PY = np.outer(p_y(x, y), cp(t))
    VDV = np.outer(u(x, y), c(t)) * np.outer(v_x(x, y), c(t)) \
        + np.outer(v(x, y), c(t)) * np.outer(v_y(x, y), c(t))
    return VT - VISCO * D2V + VDV + PY


def G_EXACT_BUBBLE(t, mesh):
    """
    Exact Y-component of forcing function with P1-Bubble interpolation.
    """
    VT = np.outer(bubble_interpolation(mesh, v), dc(t))
    D2V = np.outer(bubble_interpolation(mesh, delta_v), c(t))
    PY = np.outer(bubble_interpolation(mesh, p_y), cp(t))
    VDV = np.outer(bubble_interpolation(mesh, u), c(t)) \
        * np.outer(bubble_interpolation(mesh, v_x), c(t)) \
        + np.outer(bubble_interpolation(mesh, v), c(t)) \
        * np.outer(bubble_interpolation(mesh, v_y), c(t))
    return VT - VISCO * D2V + VDV + PY


def U_HIST(tgrid, x, y):
    """
    X-component of velocity history.
    """
    return np.outer(u(x,y), tgrid)


def U_HIST_BUBBLE(tgrid, mesh):
    """
    X-component of velocity history with P1-Bubble interpolation.
    """
    return np.outer(bubble_interpolation(mesh, u), tgrid)


def V_HIST(tgrid, x, y):
    """
    Y-component of velocity history.
    """
    return np.outer(v(x, y), tgrid)


def V_HIST_BUBBLE(tgrid, mesh):
    """
    Y-component of velocity history with P1-Bubble interpolation.
    """
    return np.outer(bubble_interpolation(mesh, v), tgrid)


def F_EXACT_DELAY(t, r, x, y):
    """
    Exact X-component of forcing function for the delayed NSE.
    """
    UT = np.outer(u(x, y), dc(t))
    D2U = np.outer(delta_u(x, y), c(t))
    PX = np.outer(p_x(x, y), cp(t))
    UDU = np.outer(u(x, y), c(t-r)) * np.outer(u_x(x, y), c(t)) \
        + np.outer(v(x, y), c(t-r)) * np.outer(u_y(x, y), c(t))
    return UT - VISCO * D2U + UDU + PX


def F_EXACT_BUBBLE_DELAY(t, r, mesh):
    """
    Exact X-component of forcing function for the delayed NSE with
    P1-Bubble interpolation.
    """
    UT = np.outer(bubble_interpolation(mesh, u), dc(t))
    D2U = np.outer(bubble_interpolation(mesh, delta_u), c(t))
    PX = np.outer(bubble_interpolation(mesh, p_x), cp(t))
    UDU = np.outer(bubble_interpolation(mesh, u), c(t-r)) \
        * np.outer(bubble_interpolation(mesh, u_x), c(t)) \
        + np.outer(bubble_interpolation(mesh, v), c(t-r)) \
        * np.outer(bubble_interpolation(mesh, u_y), c(t))
    return UT - VISCO * D2U + UDU + PX


def G_EXACT_DELAY(t, r, x, y):
    """
    Exact Y-component of forcing function for the delayed NSE.
    """
    VT = np.outer(v(x, y), dc(t))
    D2V = np.outer(delta_v(x, y), c(t))
    PY = np.outer(p_y(x, y), cp(t))
    VDV = np.outer(u(x, y), c(t-r)) * np.outer(v_x(x, y), c(t)) \
        + np.outer(v(x, y), c(t-r)) * np.outer(v_y(x, y), c(t))
    return VT - VISCO * D2V + VDV + PY


def G_EXACT_BUBBLE_DELAY(t, r, mesh):
    """
    Exact Y-component of forcing function for the delayed NSE with
    P1-Bubble interpolation.
    """
    VT = np.outer(bubble_interpolation(mesh, v), dc(t))
    D2V = np.outer(bubble_interpolation(mesh, delta_v), c(t))
    PY = np.outer(bubble_interpolation(mesh, p_y), cp(t))
    VDV = np.outer(bubble_interpolation(mesh, u), c(t-r)) \
        * np.outer(bubble_interpolation(mesh, v_x), c(t)) \
        + np.outer(bubble_interpolation(mesh, v), c(t-r)) \
        * np.outer(bubble_interpolation(mesh, v_y), c(t))
    return VT - VISCO * D2V + VDV + PY

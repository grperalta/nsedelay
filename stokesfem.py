# -*- coding: utf-8 -*-
"""
A python module for the finite element method of the Stokes equation on a
2-dimensional triangular mesh. Implementation is based on the two simplest
finite elements, the Taylor-Hood and mini-finite elements on triangles.

Gilbert Peralta
Department of Mathematics and Computer Science
University of the Philippines Baguio
Governor Pack Road, Baguio, Philippines 2600
Email: grperalta@up.edu.ph
9 July 2020
"""

from __future__ import division
from matplotlib import pyplot as plt
from scipy import sparse as sp
import numpy as np

__author__ = "Gilbert Peralta"
__copyright__ = "Copyright 2020, Gilbert Peralta"
__version__ = "1.0"
__institution__ = "University of the Philippines Baguio"
__email__ = "grperalta@up.edu.ph"
__date__ = "9 July 2020"

#-----------------------------------------------------------------------------------
# MESH CLASS
#-----------------------------------------------------------------------------------

def _refine(mesh):
    """
    Main function for the mesh refinement via bisection.

    Parameter
    ---------
    mesh : stokesfem.Mesh class
        The domain triangulation.

    Returns
    -------
    stokesfem.Mesh class
        The refined domain triangulation.
    """
    # update array of nodes
    new_node = np.append(mesh.node, mesh.edge_center(), axis=0)

    # update array of triangles
    new_cell = np.zeros((4*mesh.num_cell, 3), dtype=np.int)
    new_cell[:mesh.num_cell, :] \
        = np.vstack([mesh.cell[:, 0],
        mesh.num_node + mesh.cell_to_edge[:, 2],
        mesh.num_node + mesh.cell_to_edge[:, 1]]).T
    new_cell[mesh.num_cell:2*mesh.num_cell, :] \
        = np.vstack([mesh.cell[:, 1],
        mesh.num_node + mesh.cell_to_edge[:, 0],
        mesh.num_node + mesh.cell_to_edge[:, 2]]).T
    new_cell[2*mesh.num_cell:3*mesh.num_cell, :] \
        = np.vstack([mesh.cell[:, 2],
        mesh.num_node + mesh.cell_to_edge[:, 1],
        mesh.num_node + mesh.cell_to_edge[:, 0]]).T
    new_cell[3*mesh.num_cell:, :] \
        = np.vstack([mesh.num_node + mesh.cell_to_edge[:, 0],
        mesh.num_node + mesh.cell_to_edge[:, 1],
        mesh.num_node + mesh.cell_to_edge[:, 2]]).T

    return Mesh(new_node, new_cell)


class Mesh():
    """
    Triangular mesh class for domain subdivision.
    """

    def __init__(self, node, cell):
        """
        Class initialization.

        Attributes
        ----------
        node : numpy.ndarray
            Array consisting of the node coordinates.
        cell : numpy.ndarray
            Array consisting of node indices (connectivity) of the triangles.
        num_node : int
            Number of nodes.
        num_cell : int
            Number of cells or triangles.
        """
        self.node = node.astype(float)
        self.cell = cell.astype(int)
        self.num_node = node.shape[0]
        self.num_cell = cell.shape[0]

    def __str__(self):
        string = "Triangular mesh with {} nodes and {} cells."
        return string.format(self.num_node, self.num_cell)

    def __repr__(self):
        return self.__str__()

    def cell_center(self):
        """
        Returns the triangle barycenters.
        """
        return (self.node[self.cell[:, 0], :] + self.node[self.cell[:, 1], :]
            + self.node[self.cell[:, 2], :]) / 3.0

    def edge_center(self):
        """
        Returns the edge midpoints.
        """
        if hasattr(self, "edge"):
            pass
        else:
            self.build_data_struct()
        return 0.5 * (self.node[self.edge[:, 0], :]
            + self.node[self.edge[:, 1], :])

    def size(self):
        """
        Returns the largest edge length.
        """
        if hasattr(self, "edge"):
            pass
        else:
            self.build_data_struct()
        edge = self.node[self.edge[:, 0], :] - self.node[self.edge[:, 1], :]
        return max(np.linalg.norm(edge, axis=1))

    def build_data_struct(self):
        """
        Build the following additional attributes of the mesh:

        edge : numpy.ndarray
            Node connectivity defining the edges.
        num_edge : int
            Number of edges.
        bdy_edge : numpy.ndarray
            Indices of edges on the boundary.
        bdy_node : numpy.ndarray
            Indices of nodes on the boundary.
        cell_to_edge : numpy.ndarray
            Mapping of each cell to the indices of the cell edges.
        """
        # edge, number of edge, cell_to_edge
        edge_temp = np.hstack([self.cell[:, [1, 2]], self.cell[:, [2, 0]],
            self.cell[:, [0, 1]]]).reshape(3*self.num_cell, 2)
        edge_temp = np.sort(edge_temp)
        self.edge, index, inv_index = np.unique(edge_temp, axis=0,
            return_index=True, return_inverse=True)
        self.num_edge = self.edge.shape[0]
        self.cell_to_edge = inv_index.reshape((self.num_cell, 3))

        # edge_to_cell
        edge_temp = np.hstack([self.cell_to_edge[:, 0],
            self.cell_to_edge[:, 1], self.cell_to_edge[:, 2]])
        cell_temp = np.tile(np.arange(self.num_cell), (1, 3))[0]
        edge_frst, index_edge_frst = np.unique(edge_temp,
            return_index=True)
        edge_last, index_edge_last = np.unique(edge_temp[::-1],
            return_index=True)
        index_edge_last = edge_temp.shape[0] - index_edge_last - 1
        edge_to_cell = np.zeros((self.num_edge, 2), dtype=np.int64)
        edge_to_cell[edge_frst, 0] = cell_temp[index_edge_frst]
        edge_to_cell[edge_last, 1] = cell_temp[index_edge_last]
        edge_to_cell[np.nonzero(edge_to_cell[:, 0]
            == edge_to_cell[:, 1])[0], 1] = -1

        # bdy_edge and bdy_node
        self.bdy_edge = np.nonzero(edge_to_cell[:, 1] == -1)[0]
        self.bdy_node = np.unique(self.edge[self.bdy_edge])

    def int_node(self):
        """
        Array of interior nodes.
        """
        if hasattr(self, "bdy_node"):
            pass
        else:
            self.build_data_struct()
        in_node \
            = set(range(self.num_node)).difference(set(self.bdy_node))
        return np.asarray(list(in_node))

    def int_edge(self):
        """
        Array of interior edges.
        """
        if hasattr(self, "bdy_edge"):
            pass
        else:
            self.build_data_struct()
        in_edge \
            = set(range(self.num_edge)).difference(set(self.bdy_edge))
        return np.asarray(list(in_edge))

    def plot(self, show=True, **kwargs):
        """
        Plot of the mesh.

        Returns
        -------
        matplotlib.lines.Line2D object
        """
        plt.gca().set_aspect('equal')
        ax = plt.triplot(self.node[:, 0], self.node[:, 1],
            self.cell, lw=0.5, color='black', **kwargs)
        plt.box(False)
        plt.axis('off')
        plt.title(self, fontsize=11)
        if show:
            plt.show()
        return ax

    def refine(self, level=1):
        """
        Refines the triangulation by bisection method.

        Parameter
        ---------
        level : int
            Number of iterations for bisection.

        Returns
        -------
        stokesfem.Mesh class
            The refined domain triangulation.
        """
        ref_mesh = self
        try:
            for it in range(level):
                ref_mesh = _refine(ref_mesh)
        except AttributeError:
            self.build_data_struct()
            for it in range(level):
                ref_mesh = _refine(ref_mesh)
        return ref_mesh

    def all_node(self, name='taylor_hood'):
        """
        Returns all nodes for interpolation in the Taylor-Hood and
        P1-bubble methods.

        Parameter
        ---------
        name : str
            Either 'taylor_hood' or 'p1_bubble'.
        """
        if name is 'taylor_hood':
            return np.append(self.node, self.edge_center(), axis=0)
        elif name is 'p1_bubble':
            return np.append(self.node, self.cell_center(), axis=0)
        else:
            raise UserWarning('Invalid name!')

    def all_bdy_node(self):
        """
        Returns all boundary nodes in the Taylor-Hood method.
        """
        return np.append(self.bdy_node, self.num_node + self.bdy_edge)

    def get_num_global_dof(self, name='taylor_hood'):
        """
        Returns the number of global degrees of freedom.
        """
        if name is 'taylor_hood':
            return self.num_node + self.num_edge
        elif name is 'p1_bubble':
            return self.num_node + self.num_cell
        else:
            raise UserWarning('Invalid name!')

    def femprocess(self, name='taylor_hood'):
        """
        Processing mesh for matrix assembly in the finite element method.

        Parameter
        ---------
        name : str
            Either 'taylor_hood' or 'p1_bubble'.
        """
        if name is 'taylor_hood':
            self.build_data_struct()
            self.dof = self.get_num_global_dof('taylor_hood')
        elif name is 'p1_bubble':
            self.build_data_struct()
            self.dof = self.get_num_global_dof('p1_bubble')
        else:
            raise UserWarning('Invalid name!')
        return self


def rectangle_uniform_trimesh(base, height, n, m):
    """
    Generates a uniform triangulation of the rectangle having vertices at
    (0, 0), (base, 0), (0, height) and (base, height), with n and m subdivisions
    on the horizontal and vertical edges, respectively.

    Returns
    -------
    stokesfem.Mesh class
    """
    # number of elements
    numelem = 2 * (n - 1) * (m - 1)

    # pre-allocation of node array
    node = np.zeros((n * m, 2)).astype(float)

    # generation of node array
    for i in range(n):
        for j in range(m):
            # node index
            index = i * m + j
            # x-coordinates of a node
            node[index, 0] = i * base / (n - 1)
            # y-coordinate of a node
            node[index, 1] = j * height / (m - 1)

    # pre-allocation of node connectivity
    cell = np.zeros((numelem, 3)).astype(int)
    ctr = 0

    for i in range(n - 1):
        for j in range(m - 1):
            # lower right node of the rectangle determined by two intersecting
            # triangles
            lr_node = i * m + j + 1
            # lower left triangle
            cell[ctr, :] = [lr_node, lr_node + m, lr_node - 1]
            # upper right triangle
            cell[ctr + 1, :] = [lr_node + m - 1, lr_node - 1, lr_node + m]
            ctr += 2

    return Mesh(node, cell)


def square_uniform_trimesh(n):
    """
    Generates a uniform triangulation of the unit square with vertices at
    (0, 0), (1, 0), (0, 1) and (1, 1), with n subdivisions on a side.

    Returns
    -------
    stokesfem.Mesh class
    """
    return rectangle_uniform_trimesh(1.0, 1.0, n, n)


#-----------------------------------------------------------------------------------
# BASIS CLASS
#-----------------------------------------------------------------------------------

class Basis():
    """
    Class for finite element basis.
    """

    def __init__(self, val, grad, dof, dim):
        """
        Class initialization.

        Attributes
        ----------
        val : numpy.ndarray
            Function values of basis functions at the quadrature nodes with
            shape = (no. of basis elements) x (no. quadrature points)
        grad : numpy.ndarray
            Gradient values of basis functions at the quadrature nodes with
            shape = (no. of basis elements) x (no. quadrature points) x (dim)
        dof : int
            Number of local nodal degrees of freedom.
        dim : int
            Dimension of the element.
        """
        self.val = val
        self.grad = grad
        self.dof = dof
        self.dim = dim


def p1basis(p):
    """
    P1 Lagrange finite element bases evaluated at the points p.

    Returns
    -------
    stokesfem.Basis class
    """
    dof, dim = 3, 2
    x, y = p[:, 0], p[:, 1]
    numnode = p.shape[0]

    val = np.zeros((3, numnode)).astype(float)
    grad = np.zeros((3, numnode, 2)).astype(float)
    one = np.ones(numnode).astype(float)
    zero = np.zeros(numnode).astype(float)

    # basis function values
    val[0, :] = 1 - x - y
    val[1, :] = x
    val[2, :] = y

    # gradient values of basis functions
    grad[0, :, :] = np.array([-one, -one]).T
    grad[1, :, :] = np.array([ one, zero]).T
    grad[2, :, :] = np.array([zero,  one]).T

    return Basis(val, grad, dof, dim)


def p1bubblebasis(p):
    """
    P1-Bubble Lagrange finite element bases evaluated at the points p.

    Returns
    -------
    stokesfem.Basis class
    """
    dof, dim = 4, 2
    x, y = p[:, 0], p[:, 1]
    numnode = p.shape[0]

    val = np.zeros((4, numnode)).astype(float)
    grad = np.zeros((4, numnode, 2)).astype(float)
    one = np.ones(numnode).astype(float)
    zero = np.zeros(numnode).astype(float)

    # basis function values
    val[0, :] = 1 - x - y
    val[1, :] = x
    val[2, :] = y
    val[3, :] = 27 * val[0, :] * val[1, :] * val[2, :]

    # gradient values of basis functions
    grad[0, :, :] = np.array([-one, -one]).T
    grad[1, :, :] = np.array([ one, zero]).T
    grad[2, :, :] = np.array([zero,  one]).T
    for k in [0, 1]:
        grad[3, :, k] = 27 * (val[0, :] * val[1, :] * grad[2, :, k]
                + val[1, :] * val[2, :] * grad[0, :, k]
                + val[2, :] * val[0, :] * grad[1, :, k] )

    return Basis(val, grad, dof, dim)


def p2basis(p):
    """
    P2 Lagrange finite element bases evaluated at the points p.

    Returns
    -------
    stokesfem.Basis class
    """
    dof, dim = 6, 2
    x, y = p[:, 0], p[:, 1]
    numnode = p.shape[0]

    val = np.zeros((6, numnode)).astype(float)
    grad = np.zeros((6, numnode, 2)).astype(float)
    zero = np.zeros(numnode).astype(float)

    # basis function values
    val[0, :] = (1 - x - y) * (1 - 2 * x - 2 * y)
    val[1, :] = x * (2 * x - 1)
    val[2, :] = y * (2 * y - 1)
    val[3, :] = 4 * x * y
    val[4, :] = 4 * y * (1 - x - y)
    val[5, :] = 4 * x * (1 - x - y)

    # gradient values of basis functions
    grad[0, :, :] = np.array([4 * x + 4 * y - 3, 4 * x + 4 * y - 3]).T
    grad[1, :, :] = np.array([4 * x - 1, zero]).T
    grad[2, :, :] = np.array([zero, 4 * y - 1]).T
    grad[3, :, :] = np.array([4 * y, 4 * x]).T
    grad[4, :, :] = np.array([-4 * y, 4 * (1 - x - 2 * y)]).T
    grad[5, :, :] = np.array([4 * (1 - 2 * x - y), - 4 * x]).T

    return Basis(val, grad, dof, dim)


def bubble_interpolation(mesh, fun):
    """
    Calculate the coefficients for the bubble basis functions.

    Parameters
    ----------
    mesh : stokesfem.Mesh class
        Triangulation of the domain.
    fun : callable
        Function to be interpolated.

    Returns
    -------
    numpy.ndarray
        Nodal values of the interpolated function.
    """
    f_bub = np.zeros(mesh.num_cell).astype(float)

    center = mesh.cell_center()
    x_bar, y_bar = center[:, 0], center[:, 1]

    for i in range(mesh.num_cell):
        x_tri = mesh.node[mesh.cell[i, :], 0]
        y_tri = mesh.node[mesh.cell[i, :], 1]
        Acoeff = np.array([[x_tri[0], y_tri[0], 1],
            [x_tri[1], y_tri[1], 1], [x_tri[2], y_tri[2], 1]])
        coef = np.linalg.solve(Acoeff, np.identity(3))
        coef = np.dot(np.array([x_bar[i], y_bar[i], 1]), coef)
        f_bub[i] = fun(x_bar[i], y_bar[i]) - np.dot(coef, fun(x_tri, y_tri))

    return np.append(fun(mesh.node[:, 0], mesh.node[:, 1]), f_bub, axis=0)

#-----------------------------------------------------------------------------------
# QUADRATURE CLASS
#-----------------------------------------------------------------------------------

class Quadrature():
    """
    Class for numerical quadrature formulas.
    """

    def __init__(self, node, weight, dim, order):
        """
        Class initialization.

        Attributes
        ----------
        node : numpy.ndarray
            Quadrature nodes.
        weight : numpy.array
            Quadrature weights.
        dim : int
            Dimension of integration.
        order : int
            Order of quadrature rule.
        """
        self.node = node
        self.weight = weight
        self.order = order
        self.dim = dim


def quad_gauss_tri(order):
    """
    Gauss integration on the reference triangle with vertices at
    (0, 0), (0, 1) and (1, 0).

    Parameters
    ----------
    order : int
        Order of Gaussian quadrature.

    Returns
    -------
    stokesfem.Quadrature class

    To do
    ------
    Include quadrature order higher than 6.
    """

    dim = 2

    if order == 1:
        node = np.array([1./3, 1./3])
        weight = np.array([1./2])
    elif order == 2:
        node = np.array([
            [0.1666666666666666666666, 0.6666666666666666666666,
             0.1666666666666666666666],
            [0.1666666666666666666666, 0.1666666666666666666666,
             0.6666666666666666666666]]).T
        weight = np.array(
            [0.1666666666666666666666,
             0.1666666666666666666666,
             0.1666666666666666666666])
    elif order == 3:
        node = np.array([
            [0.333333333333333, 0.200000000000000,
             0.600000000000000, 0.200000000000000],
            [0.333333333333333, 0.600000000000000,
             0.200000000000000, 0.200000000000000]]).T
        weight = np.array(
            [-0.28125000000000, 0.260416666666667,
             0.260416666666667, 0.260416666666667])
    elif order == 4:
        node = np.array([
            [0.4459484909159650, 0.0915762135097699, 0.1081030181680700,
             0.8168475729804590, 0.4459484909159650, 0.0915762135097710],
            [0.1081030181680700, 0.8168475729804590, 0.4459484909159650,
             0.0915762135097710, 0.4459484909159650, 0.0915762135097699]]).T
        weight = np.array(
            [0.111690794839006, 0.054975871827661, 0.111690794839006,
             0.054975871827661, 0.111690794839006, 0.054975871827661])
    elif order == 5:
        node = np.array([
            [0.333333333333333, 0.470142064105115, 0.101286507323457,
             0.059715871789770, 0.797426985353087, 0.470142064105115,
             0.101286507323456],
            [0.333333333333333, 0.059715871789770, 0.797426985353087,
             0.470142064105115, 0.101286507323456, 0.470142064105115,
             0.101286507323457]]).T
        weight = np.array(
            [0.1125000000000000, 0.0661970763942530, 0.0629695902724135,
             0.0661970763942530, 0.0629695902724135, 0.0661970763942530,
             0.0629695902724135])
    elif order == 6:
        node = np.array([
            [0.2492867451709110, 0.0630890144915021, 0.5014265096581790,
             0.8738219710169960, 0.2492867451709100, 0.0630890144915020,
             0.6365024991213990, 0.3103524510337840, 0.0531450498448170,
             0.0531450498448170, 0.3103524510337840, 0.6365024991213990],
            [0.5014265096581790, 0.8738219710169960, 0.2492867451709100,
             0.0630890144915020, 0.2492867451709110, 0.0630890144915021,
             0.0531450498448170, 0.0531450498448170, 0.3103524510337840,
             0.6365024991213990, 0.6365024991213990, 0.3103524510337840]]).T
        weight = np.array(
            [0.0583931378631895, 0.0254224531851035, 0.0583931378631895,
             0.0254224531851035, 0.0583931378631895, 0.0254224531851035,
             0.0414255378091870, 0.0414255378091870, 0.0414255378091870,
             0.0414255378091870, 0.0414255378091870, 0.0414255378091870])
    else:
        node = None
        weight = None
        order = None
        dim = None
        message = 'Number of quadrature order available up to 6 only.'
        raise UserWarning(message)

    return Quadrature(node, weight, dim, order)

#-----------------------------------------------------------------------------------
# TRANSFORMATION CLASS
#-----------------------------------------------------------------------------------

class Transform():
    """
    Class for transformations from the reference element to the physical element.
    """

    def __init__(self):
        """
        Class initialization.

        Parameters
        ----------
        invmatT : numpy.ndarray
            Inverse transpose of the transformation matrices with
            shape = (no. of cell) x 2 x 2.
        det : numpy.ndarray
            Absolute value of determinants of the transformation matrices with
            length = (no. of cell).
        """
        self.invmatT = None
        self.det = None


def affine_transform(mesh):
    """
    Generates the affine transformations Tx = Ax + b from the reference triangle
    with vertices at (0, 0), (0, 1) and (1, 0) to each element of the mesh.

    Parameter
    ---------
    mesh : stokesfem.Mesh class
        The domain triangulation.

    Returns
    -------
    stokesfem.Transform class
    """
    transform = Transform()
    transform.invmatT = np.zeros((mesh.num_cell, 2, 2)).astype(float)

    # coordinates of the triangles with local indices 0, 1, 2
    A = mesh.node[mesh.cell[:, 0], :]
    B = mesh.node[mesh.cell[:, 1], :]
    C = mesh.node[mesh.cell[:, 2], :]

    a = B - A
    b = C - A

    transform.invmatT[:, 0, :] = a
    transform.invmatT[:, 1, :] = b
    transform.det = np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])
    transform.invmatT = np.linalg.inv(transform.invmatT)

    return transform

#-----------------------------------------------------------------------------------
# MATRIX ASSEMBLY
#-----------------------------------------------------------------------------------

class FEMDataStruct():
    """
    Class for finite element data structure.
    """

    def __init__(self, name, quad, vbasis, dxvbasis, dyvbasis, pbasis,
        transform, vbasis_localdof, pbasis_localdof):
        """
        Class initialization.

        Parameters
        ----------
        name : str
            Name of the finite element.
        quad : stokesfem.Quadrature class
            Numerical quadrature data structure.
        vbasis : numpy.ndarray
            Velocity basis functions at quadrature nodes.
        dxvbasis : numpy.ndarray
            Derivative wrt x of velocity basis functions at quadrature nodes.
        dyvbasis : numpy.ndarray
            Derivative wrt y of velocity basis functions at quadrature nodes.
        pbasis : numpy.ndarray
            Pressure basis functions at quadrature nodes.
        transform : stokesfem.Transform class
            Affine transformations data structure.
        vbasis_localdof : int
            Number of local degrees of freedom for velocity.
        pbasis_localdof : int
            Number of local degrees of freedom for pressure.
        """

        self.name = name
        self.quad = quad
        self.vbasis = vbasis
        self.dxvbasis = dxvbasis
        self.dyvbasis = dyvbasis
        self.pbasis = pbasis
        self.transform = transform
        self.vbasis_localdof = vbasis_localdof
        self.pbasis_localdof = pbasis_localdof

    def __str__(self):
        return "{} FEM".format(self.name.upper())


def get_fem_data_struct(mesh, quad_order=6, name='taylor_hood'):
    """
    Returns the finite element data structure for matrix assembly.

    Parameters
    ----------
    mesh : stokesfem.Mesh class
        Triangulation of the domain.
    quad_order : int
        Order of numerical quadrature.
    name : str
        Either 'taylor_hood' or 'p1_bubble'.

    Returns
    -------
    stokesfem.FEMDataStruct class
    """
    quad = quad_gauss_tri(quad_order)

    if name is 'taylor_hood':
        vbasis = p2basis(quad.node)
    elif name is 'p1_bubble':
        vbasis = p1bubblebasis(quad.node)
    else:
        raise UserWarning('Invalid name!')
    pbasis = p1basis(quad.node)

    # Affine transformatons from reference triangle to physical triangle
    transform = affine_transform(mesh)

    # Transformed gradient at reference triangle
    gradvbasis = np.zeros((mesh.num_cell, vbasis.dim, vbasis.dof,
        len(quad.node))).astype(float)
    for gpt in range(len(quad.node)):
        gradvbasis_temp = vbasis.grad[:, gpt, :]
        gradvbasis[:, :, :, gpt] = \
            np.dot(transform.invmatT.reshape(vbasis.dim*mesh.num_cell, vbasis.dim),
            gradvbasis_temp.T).reshape(mesh.num_cell, vbasis.dim,
            vbasis.dof)

    return FEMDataStruct(name, quad, vbasis.val, gradvbasis[:, 0, :, :],
        gradvbasis[:, 1, :, :], pbasis.val, transform, vbasis.dof,
        pbasis.dof)


def local_to_global(mesh, j, name='taylor_hood'):
    """
    Returns the array for local to global dof mapping.
    """
    if j < 3:
        return mesh.cell[:, j]
    else:
        if name is 'taylor_hood':
            return mesh.num_node + mesh.cell_to_edge[:, j-3]
        elif name is 'p1_bubble':
            return mesh.num_node + np.array(range(mesh.num_cell))
        else:
            raise UserWarning('Invalid name!')


def assemble(mesh, femdatastruct):
    """
    Matrix assembly.

    Parameters
    ----------
    mesh : stokesfem.Mesh class
        Triangulation of the domain.
    femdatastruct : stokesfem.FEMDataStruct class
        Finite element data structure.

    Returns
    -------
    A, M, Bx, By, Mp : tuple of scipy.sparse.csr_matrix
        stiffness matrix A, mass matrix M, components [Bx, By] of the discrete
        divergence matrix, and pressure mass matrix Mp
    """
    # pre-allocation of arrays for matrix entries
    Ae = np.zeros((mesh.num_cell, femdatastruct.vbasis_localdof,
        femdatastruct.vbasis_localdof)).astype(float)
    Me = np.zeros((mesh.num_cell, femdatastruct.vbasis_localdof,
        femdatastruct.vbasis_localdof)).astype(float)
    Bxe = np.zeros((mesh.num_cell, femdatastruct.pbasis_localdof,
        femdatastruct.vbasis_localdof)).astype(float)
    Bye = np.zeros((mesh.num_cell, femdatastruct.pbasis_localdof,
        femdatastruct.vbasis_localdof)).astype(float)
    Mpe = np.zeros((mesh.num_cell, femdatastruct.pbasis_localdof,
        femdatastruct.pbasis_localdof)).astype(float)

    # element contributions
    for gpt in range(len(femdatastruct.quad.weight)):
        wgpt = femdatastruct.quad.weight[gpt]
        phi = femdatastruct.vbasis[:, gpt]
        dphidx = femdatastruct.dxvbasis[:, :, gpt]
        dphidy = femdatastruct.dyvbasis[:, :, gpt]
        psi = femdatastruct.pbasis[:, gpt]
        for j in range(femdatastruct.vbasis_localdof):
            for k in range(femdatastruct.vbasis_localdof):
                Me[:, j, k] += wgpt * femdatastruct.transform.det \
                    * phi[j] * phi[k]
                Ae[:, j, k] += wgpt * femdatastruct.transform.det \
                    * dphidx[:, j] * dphidx[:, k]
                Ae[:, j, k] += wgpt * femdatastruct.transform.det \
                    * dphidy[:, j] * dphidy[:, k]
            for k in range(femdatastruct.pbasis_localdof):
                Bxe[:, k, j] += wgpt * femdatastruct.transform.det \
                    * dphidx[:, j] * psi[k]
                Bye[:, k, j] += wgpt * femdatastruct.transform.det \
                    * dphidy[:, j] * psi[k]
        for j in range(femdatastruct.pbasis_localdof):
            for k in range(femdatastruct.pbasis_localdof):
                Mpe[:, j, k] += wgpt * femdatastruct.transform.det \
                    * psi[j] * psi[k]

    # pre-allocation of sparse matrices
    M = sp.csr_matrix((mesh.dof, mesh.dof))
    A = sp.csr_matrix((mesh.dof, mesh.dof))
    Bx = sp.csr_matrix((mesh.num_node, mesh.dof))
    By = sp.csr_matrix((mesh.num_node, mesh.dof))
    Mp = sp.csr_matrix((mesh.num_node, mesh.num_node))

    # sparse matrix assembly
    for j in range(femdatastruct.vbasis_localdof):
        row = local_to_global(mesh, j, femdatastruct.name)
        for k in range(femdatastruct.vbasis_localdof):
            col = local_to_global(mesh, k, femdatastruct.name)
            M += sp.coo_matrix((Me[:, j, k], (row, col)),
                    shape=(mesh.dof, mesh.dof)).tocsr()
            A += sp.coo_matrix((Ae[:, j, k], (row, col)),
                    shape=(mesh.dof, mesh.dof)).tocsr()
        for k in range(femdatastruct.pbasis_localdof):
            col = local_to_global(mesh, k, femdatastruct.name)
            Bx += sp.coo_matrix((Bxe[:, k, j], (col, row)),
                    shape=(mesh.num_node, mesh.dof)).tocsr()
            By += sp.coo_matrix((Bye[:, k, j], (col, row)),
                    shape=(mesh.num_node, mesh.dof)).tocsr()
    for j in range(femdatastruct.pbasis_localdof):
        row = local_to_global(mesh, j, femdatastruct.name)
        for k in range(femdatastruct.pbasis_localdof):
            col = local_to_global(mesh, k, femdatastruct.name)
            Mp += sp.coo_matrix((Mpe[:, j, k], (row, col)),
                    shape=(mesh.num_node, mesh.num_node)).tocsr()

    return A, M, Bx, By, Mp


def convection(mesh, u, v, femdatastruct):
    """
    Convection matrix assembly for ((V.Grad) z, w), where V = (u, v) is known
    and z is the scalar test function.

    Parameters
    ----------
    mesh : stokesfem.Mesh class
        Triangulation of the domain.
    u, v : numpy.ndarray
        Components of the velocity vector field V = (u, v).
    femdatastruct : stokesfem.FEMDataStruct class
        Finite element data structure.

    Returns
    -------
    scipy.sparse.csr_matrix
    """
    # pre-allocation of array for matrix entries
    Ne = np.zeros((mesh.num_cell, femdatastruct.vbasis_localdof,
        femdatastruct.vbasis_localdof)).astype(float)

    # element contributions
    for gpt in range(len(femdatastruct.quad.weight)):
        wgpt = femdatastruct.quad.weight[gpt]
        phi = femdatastruct.vbasis[:, gpt]
        dphidx = femdatastruct.dxvbasis[:, :, gpt]
        dphidy = femdatastruct.dyvbasis[:, :, gpt]
        u_temp = np.zeros(mesh.num_cell).astype(float)
        v_temp = np.zeros(mesh.num_cell).astype(float)
        for vrtx in range(femdatastruct.vbasis_localdof):
            u_temp += u[local_to_global(mesh, vrtx, femdatastruct.name)] \
                * phi[vrtx]
            v_temp += v[local_to_global(mesh, vrtx, femdatastruct.name)] \
                * phi[vrtx]
        for j in range(femdatastruct.vbasis_localdof):
            for k in range(femdatastruct.vbasis_localdof):
                Ne[:, j, k] += wgpt * femdatastruct.transform.det \
                    * u_temp * phi[k] * dphidx[:, j]
                Ne[:, j, k] += wgpt * femdatastruct.transform.det \
                    * v_temp * phi[k] * dphidy[:, j]

    # pre-allocation of sparse matrices
    N = sp.csr_matrix((mesh.dof, mesh.dof))

    # sparse matrix assembly
    for j in range(femdatastruct.vbasis_localdof):
        row = local_to_global(mesh, j, femdatastruct.name)
        for k in range(femdatastruct.vbasis_localdof):
            col = local_to_global(mesh, k, femdatastruct.name)
            N += sp.coo_matrix((Ne[:, j, k], (row, col)),
                shape=(mesh.dof, mesh.dof)).tocsr()
    return N


def convection_dual(mesh, u, v, femdatastruct):
    """
    Dual convection matrix assembly for ((Grad V).T W, Z), where V = (u, v)
    is known and Z is the vector test function.

    Parameters
    ----------
    mesh : stokesfem.Mesh class
        Triangulation of the domain.
    u, v : numpy.ndarray
        Components of the velocity vector field V = (u, v).
    femdatastruct : stokesfem.FEMDataStruct class
        Finite element data structure.

    Returns
    -------
    scipy.sparse.csr_matrix
    """
    # pre-allocation of arrays for matrix entries
    Ne1x = np.zeros((mesh.num_cell, femdatastruct.vbasis_localdof,
        femdatastruct.vbasis_localdof)).astype(float)
    Ne2x = np.zeros((mesh.num_cell, femdatastruct.vbasis_localdof,
        femdatastruct.vbasis_localdof)).astype(float)
    Ne1y = np.zeros((mesh.num_cell, femdatastruct.vbasis_localdof,
        femdatastruct.vbasis_localdof)).astype(float)
    Ne2y = np.zeros((mesh.num_cell, femdatastruct.vbasis_localdof,
        femdatastruct.vbasis_localdof)).astype(float)

    # element contributions
    for gpt in range(len(femdatastruct.quad.weight)):
        wgpt = femdatastruct.quad.weight[gpt]
        phi = femdatastruct.vbasis[:, gpt]
        dphidx = femdatastruct.dxvbasis[:, :, gpt]
        dphidy = femdatastruct.dyvbasis[:, :, gpt]
        u_x_temp = np.zeros(mesh.num_cell).astype(float)
        u_y_temp = np.zeros(mesh.num_cell).astype(float)
        v_x_temp = np.zeros(mesh.num_cell).astype(float)
        v_y_temp = np.zeros(mesh.num_cell).astype(float)
        for vrtx in range(femdatastruct.vbasis_localdof):
            u_x_temp += u[local_to_global(mesh, vrtx, femdatastruct.name)] \
                * dphidx[:, vrtx]
            u_y_temp += u[local_to_global(mesh, vrtx, femdatastruct.name)] \
                * dphidy[:, vrtx]
            v_x_temp += v[local_to_global(mesh, vrtx, femdatastruct.name)] \
                * dphidx[:, vrtx]
            v_y_temp += v[local_to_global(mesh, vrtx, femdatastruct.name)] \
                * dphidy[:, vrtx]
        for j in range(femdatastruct.vbasis_localdof):
            for k in range(femdatastruct.vbasis_localdof):
                Ne1x[:, j, k] += wgpt * femdatastruct.transform.det \
                    * u_x_temp * phi[j] * phi[k]
                Ne2x[:, j, k] += wgpt * femdatastruct.transform.det \
                    * u_y_temp * phi[j] * phi[k]
                Ne1y[:, j, k] += wgpt * femdatastruct.transform.det \
                    * v_x_temp * phi[j] * phi[k]
                Ne2y[:, j, k] += wgpt * femdatastruct.transform.det \
                    * v_y_temp * phi[j] * phi[k]

    # pre-allocation of sparse matrix
    N = sp.csr_matrix((2*mesh.dof, 2*mesh.dof))

    # sparse matrix assembly
    for j in range(femdatastruct.vbasis_localdof):
        row = local_to_global(mesh, j, femdatastruct.name)
        for k in range(femdatastruct.vbasis_localdof):
            col = local_to_global(mesh, k, femdatastruct.name)
            N += sp.coo_matrix((Ne1x[:, j, k], (row, col)),
                shape=(2*mesh.dof, 2*mesh.dof)).tocsr()
            N += sp.coo_matrix((Ne2x[:, j, k], (row, mesh.dof+col)),
                shape=(2*mesh.dof, 2*mesh.dof)).tocsr()
            N += sp.coo_matrix((Ne1y[:, j, k], (mesh.dof+row, col)),
                shape=(2*mesh.dof, 2*mesh.dof)).tocsr()
            N += sp.coo_matrix((Ne2y[:, j, k], (mesh.dof+row, mesh.dof+col)),
                shape=(2*mesh.dof, 2*mesh.dof)).tocsr()
    return N

def vorticity(mesh, femdatastruct):
    """
    Matrix assembly for vorticity (Curl V, Curl Z), where Z is the test function.

    Parameters
    ----------
    mesh : stokesfem.Mesh class
        Triangulation of the domain.
    femdatastruct : stokesfem.FEMDataStruct class
        Finite element data structure.

    Returns
    -------
    scipy.sparse.csr_matrix
    """
    # pre-allocation of arrays for matrix entries
    Vxxe = np.zeros((mesh.num_cell, femdatastruct.vbasis_localdof,
        femdatastruct.vbasis_localdof)).astype(float)
    Vxye = np.zeros((mesh.num_cell, femdatastruct.vbasis_localdof,
        femdatastruct.vbasis_localdof)).astype(float)
    Vyxe = np.zeros((mesh.num_cell, femdatastruct.vbasis_localdof,
        femdatastruct.vbasis_localdof)).astype(float)
    Vyye = np.zeros((mesh.num_cell, femdatastruct.vbasis_localdof,
        femdatastruct.vbasis_localdof)).astype(float)

    # element contributions
    for gpt in range(len(femdatastruct.quad.weight)):
        wgpt = femdatastruct.quad.weight[gpt]
        dphidx = femdatastruct.dxvbasis[:, :, gpt]
        dphidy = femdatastruct.dyvbasis[:, :, gpt]
        for j in range(femdatastruct.vbasis_localdof):
            for k in range(femdatastruct.vbasis_localdof):
                Vxxe[:, j, k] += wgpt * femdatastruct.transform.det \
                    * dphidx[:, j] * dphidx[:, k]
                Vxye[:, j, k] += wgpt * femdatastruct.transform.det \
                    * dphidx[:, j] * dphidy[:, k]
                Vyxe[:, j, k] += wgpt * femdatastruct.transform.det \
                    * dphidy[:, j] * dphidx[:, k]
                Vyye[:, j, k] += wgpt * femdatastruct.transform.det \
                    * dphidy[:, j] * dphidy[:, k]

    # pre-allocation of sparse matrices
    Vxx = sp.csr_matrix((mesh.dof, mesh.dof))
    Vxy = sp.csr_matrix((mesh.dof, mesh.dof))
    Vyx = sp.csr_matrix((mesh.dof, mesh.dof))
    Vyy = sp.csr_matrix((mesh.dof, mesh.dof))

    # sparse matrix assembly
    for j in range(femdatastruct.vbasis_localdof):
        row = local_to_global(mesh, j, femdatastruct.name)
        for k in range(femdatastruct.vbasis_localdof):
            col = local_to_global(mesh, k, femdatastruct.name)
            Vxx += sp.coo_matrix((Vxxe[:, j, k], (row, col)),
                    shape=(mesh.dof, mesh.dof)).tocsr()
            Vxy += sp.coo_matrix((Vxye[:, j, k], (row, col)),
                    shape=(mesh.dof, mesh.dof)).tocsr()
            Vyx += sp.coo_matrix((Vyxe[:, j, k], (row, col)),
                    shape=(mesh.dof, mesh.dof)).tocsr()
            Vyy += sp.coo_matrix((Vyye[:, j, k], (row, col)),
                    shape=(mesh.dof, mesh.dof)).tocsr()
    V = sp.bmat([[Vyy, -Vxy], [-Vyx, Vxx]], format='csr')
    return V


def apply_noslip_bc(A, no_slip_node_index):
    """
    Apply no-slip boundary conditions to the sparse matrix A and convert it to
    CSC format.

    Returns
    -------
    scipy.sparse.csc_matrix
    """
    A = A.tolil()
    if type(no_slip_node_index) is not list:
        no_slip_node_index = list(no_slip_node_index)
    A[no_slip_node_index, :] = 0.0
    A[:, no_slip_node_index] = 0.0
    for k in list(no_slip_node_index):
        A[k, k] = 1.0
    return A.tocsc()

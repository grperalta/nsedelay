# nsedelay

**FILE DESCRIPTIONS**

``demo_nse.py``
Runs test for the implicit-explicit (IMEX) Euler scheme for the Navier-Stokes
equation using the Taylor-Hood and Mini-Finite elements.


``demo_nsedelay.py``
Runs test for the IMEX Euler scheme for the Navier-Stokes equation with delay
in the convection term using the Taylor-Hood and Mini-Finite elements.


``demo_stokes.py``
Runs tests for the matrix assemblies in the module ``stokesfem.py``.


``functions.py``
The functions and their derivatives used in the tests.


``nsedelay.py``
This Python script approximates the solution of the following optimal control
problem for the Navier-Stokes equation with delay in the convection term.
The problem is given by:

$$
	\left\{
	\begin{aligned}
	& \min_{q \in L^2(0, T; L^2(\Omega)^2)}
	\frac{\alpha_{\Omega_T}}{2}\int_0^T \!\!\!\int_\Omega
	|u - u_d|^2 dx dt
	+ \frac{\alpha_T}{2}\int_{\Omega} |u(T) - u_{T}|^2 dx \\
	&\hspace{0.8in} + \frac{\alpha_R}{2}\int_0^T \!\!\!\int_\Omega
	|\nabla \times u|^2 dx dt +
	\frac{\alpha}{2}
	\int_0^T \!\!\!\int_\Omega |q|^2 dx dt\\
	&\text{subject to the state equation}\\
	&\qquad
	\begin{aligned}
	\partial_tu - \nu\Delta u + \mathrm{div} (u^r \otimes u) + \nabla p
	&= f + q && \text{ in } (0, T) \times \Omega,\\
	 \mathrm{div}\, u &= 0 && \text{ in } (-r, T) \times \Omega,\\
	u &= 0 && \text{ on } (0, T) \times \Gamma,\\
	u(0) &= u_0 && \text{ in } \Omega,\\
	u &= z && \text{ in } \Omega_r := (-r, 0) \times \Omega.\\
	\end{aligned}
	\end{aligned}
	\right.
$$

Here, $u$, $p$, $u_d$, $u_T$ and $q$ are the fluid velocity, fluid pressure,
desired velocity, desired velocity at the terminal time and control, respectively.
The default domain $\Omega$ is the unit square $(0, 1)^2$. Spatial discretization
is based on the Taylor-Hood (P2/P1) or P1Bubble/P1 triangular elements, while
for the time discretization, an IMEX Euler scheme is utilized. In the case where
there is no delay ($r = 0$), the scheme reduces to the usual Navier-Stokes equation.

The three standard python packages NumPy, SciPy and Matplotlib are required in
the implementation. Matrices for the finite element assembly are obtained using
the accompanied module ``stokesfem.py``.


``optim_eoc.py``
Runs tests for the order of convergences for the optimal control of the delayed
Navier-Stokes equation using the IMEX Euler scheme. Output written in
``optim_eoc.txt``.


``optim_eoc_thikonov.py``
Runs tests for the order of convergences with respect to the Tikhonov
regularization. Output written in ``optim_eoc_thikonov.txt``.


``optim_velocity.py``
Implements the velocity tracking problem for the optimal control of the delayed
Navier-Stokes equation using the IMEX Euler scheme. Output written in
``optim_velocity.txt``


``optim_vorticity.py``
Implements the vorticity minimization problem for the optimal control of the
delayed Navier-Stokes equation using the IMEX Euler scheme. Output written in
``optim_vorticity.txt``


``plot_optim_velocity.py``
Plots the results of the ``optim_velocity.py`` script.


``plot_optim_vorticity.py``
Plots the results of the ``optim_vorticity.py`` script.


``stokesfem.py``
A python module for the finite element method of the Stokes equation on a
2-dimensional triangular mesh. Implementation is based on the two simplest
finite elements, the Taylor-Hood and mini-finite elements on triangles.


``utils.py``
Miscellaneous functions.


``init_data.npy``
File for the initial data.

You may view this Markdown file better at https://stackedit.io/editor

If you find these codes useful, you can cite the manuscript as:
*G. Peralta and J. S. Simon, Optimal Control for the Navier-Stokes
Equation with Time Delay in the Convection: Analysis and Finite Element
Approximations, and John Sebastian Simon, to appear in Journal of
Mathematical Fluid Mechanics.*

Gilbert Peralta
Department of Mathematics and Computer Science
University of the Philippines Baguio
Governor Pack Road, Baguio, Philippines 2600
Email: grperalta@up.edu.ph
7 January 2021

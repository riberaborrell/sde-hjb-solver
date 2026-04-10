import os
import time
from typing import Any, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg

from sde_hjb_solver.utils_path import save_data, load_data
import sde_hjb_solver.figures

class SolverHJB2D:
    """Finite-difference solver for the 2D HJB boundary value problem.

    Solves:
        0 = LΨ − f Ψ in S
        Ψ = exp(− g) in ∂S

    where L is the infinitesimal generator of the uncontrolled 2D diffusion
    process and f, g are running and terminal costs.

    Attributes:
        sde: Controlled SDE instance defining drift/diffusion and costs.
        h: Spatial grid step size.
        ct_initial: Start time for solver timing.
        ct_final: End time for solver timing.
        ct: Total elapsed compute time.
        psi: Solution of the boundary value problem.
        solved: Whether the solver has produced a solution.
        value_function: Value function (phi = -log(psi)).
        u_opt: Optimal control computed from the value function.
        mfht: Mean first hitting time estimate (if computed).
    """

    def __init__(self, sde: Any, h: float, load: bool = False) -> None:
        """Initialize the 2D solver.

        Args:
            sde: Controlled SDE instance (2D).
            h: Grid step size.
            load: Whether to load a previously saved solution.

        Raises:
            NotImplementedError: If sde.d != 2.
        """

        if sde.d != 2:
            raise NotImplementedError('d > 2 not supported')

        # sde object
        self.sde = sde

        # discretization step
        self.h = h

        # rel directory path
        self.rel_dir_path = os.path.join(sde.__str__(), 'h{:.0e}'.format(h))

        if load:
            self.load()

    def start_timer(self) -> None:
        """Start the computation timer."""
        self.ct_initial = time.perf_counter()

    def stop_timer(self) -> None:
        """Stop the computation timer."""
        self.ct_final = time.perf_counter()
        self.ct = self.ct_final - self.ct_initial

    def get_flatten_index(self, idx: tuple) -> int:
        """Map a bumpy index to a flatten index.

        Args:
            idx: Tuple of axis indices.

        Returns:
            Flatten index of the node.
        """
        assert type(idx) == tuple, 'idx must be a tuple of axis indices'
        assert len(idx) == self.sde.d, f'idx must have length {self.sde.d}'

        k = 0
        for i in range(self.sde.d):
            assert 0 <= idx[i] <= self.sde.Nx[i] - 1, (
                f'idx[{i}] must be in [0, {self.sde.Nx[i] - 1}]'
            )
            Nx_prod = 1
            for j in range(i+1, self.sde.d):
                Nx_prod *= self.sde.Nx[j]
            k += idx[i] * Nx_prod

        return k

    def get_bumpy_index(self, k: int) -> tuple:
        """Map a flatten index to a bumpy (axis) index.

        Args:
            k: Flatten index of the node.

        Returns:
            Tuple of axis indices.
        """
        #assert type(k) == int, ''
        assert 0 <= k <= self.sde.Nh - 1, (
            f'k must be in [0, {self.sde.Nh - 1}]'
        )

        idx = [None for i in range(self.sde.d)]
        for i in range(self.sde.d):
            Nx_prod = 1
            for j in range(i+1, self.sde.d):
                Nx_prod *= self.sde.Nx[j]
            idx[i] = k // Nx_prod
            k -= idx[i] * Nx_prod
        return tuple(idx)

    def get_x(self, k: int) -> np.ndarray:
        """Return the coordinate of node k.

        Args:
            k: Flatten index of the node.

        Returns:
            Point in the domain.
        """
        assert k in np.arange(self.sde.Nh), (
            f'k must be a valid node index in [0, {self.sde.Nh - 1}]'
        )

        return self.sde.domain_h.reshape(self.sde.Nh, self.sde.d)[k]

    def get_flatten_idx_from_axis_neighbours(
        self,
        idx: tuple,
        i: int,
    ) -> Tuple[Optional[int], Optional[int]]:
        """Get flatten indices of neighbors along axis i.

        Args:
            idx: Tuple of axis indices.
            i: Axis index.

        Returns:
            Tuple of (k_left, k_right) neighbor indices.
        """

        # find flatten index of left neighbour wrt the i axis
        if idx[i] == 0:
            k_left = None
        else:
            left_idx = list(idx)
            left_idx[i] = idx[i] - 1
            k_left = self.get_flatten_index(tuple(left_idx))

        # find flatten index of right neighbour wrt the i axis
        if idx[i] == self.sde.Nx[i] - 1:
            k_right = None
        else:
            right_idx = list(idx)
            right_idx[i] = idx[i] + 1
            k_right = self.get_flatten_index(tuple(right_idx))

        return (k_left, k_right)

    def solve_bvp(self) -> None:
        """Solve the boundary value problem using finite differences."""

        # start timer
        self.start_timer()

        # discretized step
        h = self.h

        # discretize domain
        self.sde.discretize_domain_2d(h)

        # assemble linear system of equations: A \Psi = b.
        A = sparse.lil_matrix((self.sde.Nh, self.sde.Nh))
        b = np.zeros(self.sde.Nh)

        for k in np.arange(self.sde.Nh):


            # get discretized domain index
            idx = self.get_bumpy_index(k)

            # get point
            #x = self.get_x(idx)
            x = self.get_x(k)

            # assemble matrix A and vector b on S
            if k not in self.sde.ts_idx and k not in self.sde.boundary_idx:

                # drift and diffusion at x
                drift = self.sde.drift(x)
                sigma = self.sde.diffusion

                A[k, k] = - (sigma**2 * self.sde.d) / h**2 - self.sde.f(x)

                # x-axis neighbours
                k_left, k_right = self.get_flatten_idx_from_axis_neighbours(idx, i=0)
                A[k, k_left] = sigma**2 / (2 * h**2) - drift[0] / (2 * h)
                A[k, k_right] = sigma**2 / (2 * h**2) + drift[0] / (2 * h)

                # y-axis neighbours
                k_left, k_right = self.get_flatten_idx_from_axis_neighbours(idx, i=1)
                A[k, k_left] = sigma**2 / (2 * h**2) - drift[1] / (2 * h)
                A[k, k_right] = sigma**2 / (2 * h**2) + drift[1] / (2 * h)

            # impose condition on ∂S
            elif k in self.sde.ts_idx and not k in self.sde.boundary_idx:
                A[k, k] = 1
                b[k] = np.exp(- self.sde.g(x).item())

            # stability condition on the boundary: Psi should be flat
            elif k in self.sde.boundary_idx:
                neighbour_counter = 0

                if k in self.sde.boundary_x_idx:

                    # add neighbour
                    k_left, k_right = self.get_flatten_idx_from_axis_neighbours(idx, i=0)
                    if k_left is not None:
                        A[k, k_left] = - 1
                    elif k_right is not None:
                        A[k, k_right] = - 1

                    # update counter
                    neighbour_counter += 1

                if k in self.sde.boundary_y_idx:

                    # add neighbour
                    k_left, k_right = self.get_flatten_idx_from_axis_neighbours(idx, i=1)
                    if k_left is not None:
                        A[k, k_left] = - 1
                    elif k_right is not None:
                        A[k, k_right] = - 1

                    # update counter
                    neighbour_counter += 1

                # normalize
                A[k, k] = neighbour_counter

        # solve linear system and save
        psi = linalg.spsolve(A.tocsc(), b)
        self.psi = psi.reshape(self.sde.Nx).astype(np.float32)
        self.solved = True

        # stop timer
        self.stop_timer()

    def compute_value_function(self) -> None:
        """Compute the value function (phi = -log(psi))."""
        self.value_function =  - np.log(self.psi)

    def compute_optimal_control(self) -> None:
        """Compute the optimal control using finite differences."""
        assert hasattr(self, 'value_function'), (
            'value_function must be computed before calling compute_optimal_control'
        )
        assert self.value_function.ndim == self.sde.d, (
            f'value_function must have dimension {self.sde.d}'
        )

        # diffusion term
        sigma = self.sde.diffusion

        # preallocate optimal control
        self.u_opt = np.zeros(self.sde.Nx + (self.sde.d, ), dtype=np.float32)

        for i in range(self.sde.d):

            # idx value f type
            value_f_k_plus_idx = [slice(self.sde.Nx[0]), slice(self.sde.Nx[1])]
            value_f_k_minus_idx = [slice(self.sde.Nx[0]), slice(self.sde.Nx[1])]

            # idx u type
            u_k_idx = [slice(self.sde.Nx[0]), slice(self.sde.Nx[1]), i]
            u_0_idx = [slice(self.sde.Nx[0]), slice(self.sde.Nx[1]), i]
            u_1_idx = [slice(self.sde.Nx[0]), slice(self.sde.Nx[1]), i]
            u_N_minus_idx = [slice(self.sde.Nx[0]), slice(self.sde.Nx[1]), i]
            u_N_idx = [slice(self.sde.Nx[0]), slice(self.sde.Nx[1]), i]

            value_f_k_plus_idx[i] = slice(2, self.sde.Nx[i])
            value_f_k_minus_idx[i] = slice(0, self.sde.Nx[i] - 2)
            u_k_idx[i] = slice(1, self.sde.Nx[i] - 1)
            u_0_idx[i] = 0
            u_1_idx[i] = 1
            u_N_minus_idx[i] = self.sde.Nx[i] - 2
            u_N_idx[i] = self.sde.Nx[i] - 1

            # generalized central difference
            self.u_opt[tuple(u_k_idx)] = - sigma *(
                self.value_function[tuple(value_f_k_plus_idx)]
              - self.value_function[tuple(value_f_k_minus_idx)]
            ) / (2 * self.sde.h)
            self.u_opt[tuple(u_0_idx)] = self.u_opt[tuple(u_1_idx)]
            self.u_opt[tuple(u_N_idx)] = self.u_opt[tuple(u_N_minus_idx)]


    def save(self) -> None:
        """Save solver attributes to a .npz file."""
        # create data dictionary 
        data = {
            'h': self.sde.h,
            'domain_h': self.sde.domain_h,
            'Nx': self.sde.Nx,
            'Nh': self.sde.Nh,
            'psi': self.psi,
            'value_function': self.value_function,
            'u_opt': self.u_opt,
            'ct': self.ct,
        }
        if hasattr(self, 'mfht'):
            data['mfht'] = self.mfht

        # save arrays in a npz file
        save_data(data, self.rel_dir_path)

    def load(self) -> bool:
        """Load saved arrays and set solver attributes.

        Returns:
            True if loading succeeded; False otherwise.
        """
        data = load_data(self.rel_dir_path)
        try:
            for attr_name in data.keys():

                # get attribute from data
                attr = data[attr_name]

                # Controlled SDE attribute
                if attr_name in ['h', 'domain_h', 'Nx', 'Nh']:

                    # if attribute exists check if they are the same
                    if hasattr(self.sde, attr_name):
                        assert getattr(self.sde, attr_name) == attr, (
                            f'Loaded sde.{attr_name} does not match existing value'
                        )

                    # if attribute does not exist save attribute
                    else:
                        setattr(self.sde, attr_name, attr)

                # hjb solver attribute
                else:

                    # if attribute exists check if they are the same
                    if hasattr(self, attr_name):
                        assert getattr(self, attr_name) == attr, (
                            f'Loaded {attr_name} does not match existing value'
                        )

                    # if attribute does not exist save attribute
                    else:
                        setattr(self, attr_name, attr)

        except:
            print('Attribute to load already exists and does not match')
            return False

        # compute perturbed potential and drift
        if self.sde.is_overdamped_langevin:
            self.get_perturbed_potential_and_drift()

        return True


    def coarse_solution(self, h_coarse: float) -> None:
        """Coarsen the solution by subsampling the grid."""

        assert self.h <= h_coarse, (
            f'h_coarse must be >= h (h={self.h}, h_coarse={h_coarse})'
        )

        # discretization step ratio
        k = int(h_coarse / self.h)

        self.psi = self.psi[::k, ::k]
        self.value_function = self.value_function[::k, ::k]
        self.u_opt = self.u_opt[::k, ::k]

        if self.sde.is_overdamped_langevin:
            self.V = self.V[::k, ::k]
            self.perturbed_potential = self.perturbed_potential[::k, ::k]
            self.dV = self.dV[::k, ::k]
            self.perturbed_drift = self.perturbed_drift[::k, ::k]

    def get_psi_at_x(self, x: Union[float, np.ndarray]) -> Optional[Union[float, np.ndarray]]:
        """Evaluate the solution psi at x.

        Args:
            x: Point in the domain.

        Returns:
            psi evaluated at x, or None if psi is unavailable.
        """
        # get index of x
        idx = self.sde.get_idx(x)

        # evaluate psi at x
        return self.psi[idx] if hasattr(self, 'psi') else None

    def get_value_function_at_x(
        self,
        x: Union[float, np.ndarray],
    ) -> Optional[Union[float, np.ndarray]]:
        """Evaluate the value function at x.

        Args:
            x: Point in the domain.

        Returns:
            Value function evaluated at x, or None if unavailable.
        """
        # get index of x
        idx = self.sde.get_idx(x)

        # evaluate value function at x
        return self.value_function[idx] if hasattr(self, 'value_function') else None

    def get_u_opt_at_x(
        self,
        x: Union[float, np.ndarray],
    ) -> Optional[Union[float, np.ndarray]]:
        """Evaluate the optimal control at x.

        Args:
            x: Point in the domain.

        Returns:
            Optimal control evaluated at x, or None if unavailable.
        """
        # get index of x
        idx = self.sde.get_idx(x)

        # evaluate optimal control at x
        return self.u_opt[idx] if hasattr(self, 'u_opt') else None

    def get_perturbed_potential_and_drift(self) -> None:
        """Compute potentials, gradients, and perturbed drift fields."""

        # flatten domain_h
        x = self.sde.domain_h.reshape(self.sde.Nh, self.sde.d)

        # diffusion term
        sigma = self.sde.diffusion

        # potential, bias potential and tilted potential
        self.V = self.sde.potential(x).reshape(self.sde.Nx)
        self.bias_potential = (sigma**2) * self.value_function
        self.perturbed_potential = self.V + self.bias_potential

        # gradient and tilted drift
        self.dV = self.sde.gradient(x).reshape(self.sde.domain_h.shape)
        self.perturbed_drift = - self.dV + sigma * self.u_opt


    def write_report(self, x: Union[float, np.ndarray]) -> None:
        """Print solver parameters and solution values at x.

        Args:
            x: Point in the domain.
        """
        from sde_hjb_solver.utils import get_time_in_hms

        # space discretization
        print('\n space discretization')
        print('h = {:2.4f}'.format(self.sde.h))
        print('N_h = {:d}'.format(self.sde.Nh))

        # psi, value function and control
        print('\n psi, value function and optimal control at x')
        return

        print('x: {:2.3f}'.format(x))
        psi = self.get_psi_at_x(x)
        value_f = self.get_value_function_at_x(x)
        u_opt = self.get_u_opt_at_x(x)

        if psi is not None:
            print('psi(x) = {:2.4e}'.format(psi))

        if value_f is not None:
            print('value_f(x) = {:2.4e}'.format(value_f))

        if u_opt is not None:
            print('u_opt(x): {:2.3f}'.format(u_opt))

        # maximum value of the control
        print('\n maximum value of the optimal control')

        idx_u_max = np.argmax(self.u_opt)
        x_u_max = self.get_x(idx_u_max)
        u_opt_max = self.u_opt[idx_u_max]
        print('argmax_x u_opt(x): {:2.3f}'.format(x_u_max))
        print('max_x u_opt(x): {:2.3f}'.format(u_opt_max))

        # computational time
        h, m, s = get_time_in_hms(self.ct)
        print('\nComputational time: {:d}:{:02d}:{:02.2f}\n'.format(h, m, s))

    def plot_2d_psi(
        self,
        levels: int = 10,
        isolines: bool = True,
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plot the estimated solution psi(x).

        Args:
            levels: Number of contour levels.
            isolines: Whether to draw isolines.
            xlim: Optional x-axis limits.
            ylim: Optional y-axis limits.

        Returns:
            Matplotlib figure and axes.
        """
        fig, ax = plt.subplots()
        ax.set_title(r'Estimation of $\Psi(x)$')
        ax.set_xlabel(r'$x_1$')
        ax.set_ylabel(r'$x_2$')
        ax.set_xlim(xlim) if xlim is not None else ax.set_xlim(self.sde.domain[0])
        ax.set_ylim(ylim) if ylim is not None else ax.set_ylim(self.sde.domain[1])

        # contour f
        cs = ax.contourf(
            self.sde.domain_h[:, :, 0],
            self.sde.domain_h[:, :, 1],
            self.psi,
            levels=levels,
            extend='both',
            cmap='plasma',
        )
        if isolines: ax.contour(cs, colors='k')

        # colorbar
        cbar = fig.colorbar(cs)

        return fig, ax

    def plot_2d_value_function(
        self,
        levels: int = 10,
        isolines: bool = True,
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plot the estimated value function phi(x).

        Args:
            levels: Number of contour levels.
            isolines: Whether to draw isolines.
            xlim: Optional x-axis limits.
            ylim: Optional y-axis limits.

        Returns:
            Matplotlib figure and axes.
        """
        fig, ax = plt.subplots()
        ax.set_title(r'Estimation of $\Phi(x)$')
        ax.set_xlabel(r'$x_1$')
        ax.set_ylabel(r'$x_2$')
        ax.set_xlim(xlim) if xlim is not None else ax.set_xlim(self.sde.domain[0])
        ax.set_ylim(ylim) if ylim is not None else ax.set_ylim(self.sde.domain[1])

        # contour f
        cs = ax.contourf(
            self.sde.domain_h[:, :, 0],
            self.sde.domain_h[:, :, 1],
            self.value_function,
            levels=levels,
            extend='both',
            cmap='plasma',
        )
        if isolines: ax.contour(cs, colors='k')

        # colorbar
        cbar = fig.colorbar(cs)

        return fig, ax

    def plot_2d_perturbed_potential(
        self,
        levels: int = 10,
        isolines: bool = True,
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plot the perturbed potential.

        Args:
            levels: Number of contour levels.
            isolines: Whether to draw isolines.
            xlim: Optional x-axis limits.
            ylim: Optional y-axis limits.

        Returns:
            Matplotlib figure and axes.
        """
        fig, ax = plt.subplots()
        ax.set_title(r'Perturbed potential $(U_{pot} + U_{bias})(x)$')
        ax.set_xlabel(r'$x_1$')
        ax.set_ylabel(r'$x_2$')
        ax.set_xlim(xlim) if xlim is not None else ax.set_xlim(self.sde.domain[0])
        ax.set_ylim(ylim) if ylim is not None else ax.set_ylim(self.sde.domain[1])

        # contour f
        cs = ax.contourf(
            self.sde.domain_h[:, :, 0],
            self.sde.domain_h[:, :, 1],
            self.perturbed_potential,
            levels=levels,
            extend='both',
            cmap='Blues_r',
        )
        if isolines: ax.contour(cs, colors='k')

        # colorbar
        cbar = fig.colorbar(cs)

        return fig, ax

    def plot_2d_control(
        self,
        scale: Optional[float] = None,
        width: float = 0.005,
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plot the optimal control field.

        Args:
            scale: Optional quiver scale.
            width: Quiver arrow width.
            xlim: Optional x-axis limits.
            ylim: Optional y-axis limits.

        Returns:
            Matplotlib figure and axes.
        """
        from matplotlib import colors, cm

        fig, ax = plt.subplots()
        ax.set_title(r'Optimal control $u^*(x)$')
        ax.set_xlabel(r'$x_1$')
        ax.set_ylabel(r'$x_2$')
        ax.set_xlim(xlim) if xlim is not None else ax.set_xlim(self.sde.domain[0])
        ax.set_ylim(ylim) if ylim is not None else ax.set_ylim(self.sde.domain[1])

        X = self.sde.domain_h[:, :, 0]
        Y = self.sde.domain_h[:, :, 1]
        U = self.u_opt[:, :, 0]
        V = self.u_opt[:, :, 1]

        # set colormap
        colormap = cm.get_cmap('viridis_r', 100)
        colormap = colors.ListedColormap(
            colormap(np.linspace(0.20, 0.95, 75))
        )

        # initialize norm object and make rgba array
        C = np.sqrt(U**2 + V**2)
        norm = colors.Normalize(vmin=np.min(C), vmax=np.max(C))
        sm = cm.ScalarMappable(cmap=colormap, norm=norm)

        # quiver
        quiv = ax.quiver(
            X,
            Y,
            U,
            V,
            C,
            cmap=colormap,
            angles='xy',
            scale_units='xy',
            scale=scale,
            width=width,
        )

        # colorbar
        fig.colorbar(sm, ax=ax)

        return fig, ax

    """
    def plot_2d_perturbed_drift(self):
        fig, ax = plt.subplots()
        ax.set_title(r'Perturbed drift $\nabla(V + V_{bias})(x)$')
        self.get_perturbed_potential_and_drift()
        return fig, ax
    """

    def plot_2d_mfht(
        self,
        levels: int = 10,
        isolines: bool = True,
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plot the mean first hitting time estimate.

        Args:
            levels: Number of contour levels.
            isolines: Whether to draw isolines.
            xlim: Optional x-axis limits.
            ylim: Optional y-axis limits.

        Returns:
            Matplotlib figure and axes.
        """
        fig, ax = plt.subplots()
        ax.set_title(r'Estimation of $\mathbb{E}^x[\tau]$')
        ax.set_xlabel(r'$x_1$')
        ax.set_ylabel(r'$x_2$')
        ax.set_xlim(self.sde.domain[0])
        ax.set_ylim(self.sde.domain[1])

        # contour f
        cs = ax.contourf(
            self.sde.domain_h[:, :, 0],
            self.sde.domain_h[:, :, 1],
            self.mfht,
            levels=levels,
            extend='both',
            cmap='plasma',
        )
        if isolines: ax.contour(cs, colors='k')

        # colorbar
        cbar = fig.colorbar(cs)

        return fig, ax

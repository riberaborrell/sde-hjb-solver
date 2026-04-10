import os
import time
from typing import Any, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from sde_hjb_solver.utils_path import save_data, load_data
import sde_hjb_solver.figures

class SolverHJB1D:
    """Finite-difference solver for the 1D HJB boundary value problem.

    Solves:
        0 = LΨ − f Ψ in S
        Ψ = exp(− g) in ∂S

    where L is the infinitesimal generator of the uncontrolled 1D diffusion
    process and f, g are running and terminal costs.

    Attributes:
        sde: Controlled SDE instance defining drift/diffusion and costs.
        h: Spatial grid step size.
        ct_initial: Start time for solver timer.
        ct_final: End time for solver timer.
        ct: Total elapsed computed time.
        solved: Whether the solver has produced a solution.
        psi: Solution of the boundary value problem.
        value_function: Value function (phi = -log(psi)).
        u_opt: Optimal control computed from the value function.
        mfht: Mean first hitting time estimate (if computed).
    """

    def __init__(self, sde: Any, h: float, load: bool = False) -> None:
        """Initialize the 1D solver.

        Args:
            sde: Controlled SDE instance (1D).
            h: Grid step size.
            load: Whether to load a previously saved solution.

        Raises:
            NotImplementedError: If sde.d != 1.
        """

        if sde.d != 1:
            raise NotImplementedError('d > 1 not supported')

        # sde object
        self.sde = sde

        # discretization step
        self.h = h

        # rel directory path
        self.rel_dir_path = os.path.join(sde.__str__(), 'h{:.0e}'.format(h))

        if load: self.load()

    def start_timer(self) -> None:
        """Start the computation timer."""
        self.ct_initial = time.perf_counter()

    def stop_timer(self) -> None:
        """Stop the computation timer."""
        self.ct_final = time.perf_counter()
        self.ct = self.ct_final - self.ct_initial


    def get_x(self, k: int) -> float:
        """Return the x-coordinate of node k.

        Args:
            k: Node index.

        Returns:
            Point in the domain.
        """
        assert k in np.arange(self.sde.Nh), (
            f'k must be a valid node index in [0, {self.sde.Nh - 1}]'
        )

        return self.sde.domain_h[k]

    def solve_bvp(self) -> None:
        """Solve the boundary value problem using finite differences."""

        # start timer
        self.start_timer()

        # discretized step
        h = self.h

        # discretize domain
        self.sde.discretize_domain_1d(h)

        # assemble linear system of equations: A \Psi = b.
        A = np.zeros((self.sde.Nh, self.sde.Nh))
        b = np.zeros(self.sde.Nh)

        # nodes in boundary
        boundary_idx = np.array([0, self.sde.Nh - 1])

        # nodes in target set
        ts_idx = self.sde.ts_idx

        for k in np.arange(self.sde.Nh):

            # get point
            x = self.get_x(k)

            # assemble matrix A and vector b on S
            if k not in ts_idx and k not in boundary_idx:

                # drift and diffusion at x
                drift = self.sde.drift(x)
                sigma = self.sde.diffusion

                A[k, k] = - sigma**2 / h**2 - self.sde.f(x)
                A[k, k - 1] = sigma**2 / (2 * h**2) - drift / (2 * h)
                A[k, k + 1] = sigma**2 / (2 * h**2) + drift / (2 * h)

            # impose condition on ∂S
            elif k in ts_idx:
                A[k, k] = 1
                b[k] = np.exp(- self.sde.g(x))

            # stability condition on the boundary: Psi should be flat
            elif k in boundary_idx:
                if k == 0:
                    # Psi_0 = Psi_1
                    #A[0, 0] = 1
                    #A[0, 1] = -1
                    A[k, k] = 1
                    A[k, k+1] = -1

                if k == self.sde.Nh - 1:
                    # psi_{Nh-1} = Psi_Nh
                    #A[-1, -1] = 1
                    #A[-1, -2] = -1
                    A[k, k] = 1
                    A[k, k - 1] = -1

        # solve linear system and save
        self.psi = np.linalg.solve(A, b).astype(np.float32)
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

        # central difference approximation
        # for any k in {1, ..., Nh-2}
        # u_opt(x_k) = - sigma (Phi_{k+1} - Phi_{k-1}) / 2h 

        # diffusion term
        sigma = self.sde.diffusion

        # preallocate u_opt
        self.u_opt = np.zeros(self.sde.Nh, dtype=np.float32)

        self.u_opt[1: -1] = - sigma \
                          * (self.value_function[2:] - self.value_function[:-2]) \
                          / (2 * self.sde.h)
        self.u_opt[0] = self.u_opt[1]
        self.u_opt[-1] = self.u_opt[-2]
        self.u_opt = self.u_opt.reshape(-1, self.sde.d)

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

    def unflatten_solution(self) -> None:
        """Reshape flattened solution arrays to 1D shapes."""
        self.psi = self.psi.reshape(-1, self.sde.d)
        self.value_function = self.value_function.reshape(-1, self.sde.d)
        self.u_opt = self.u_opt.reshape(-1, self.sde.d)

    def coarse_solution(self, h_coarse: float) -> None:
        """Coarsen the solution by subsampling the grid."""

        assert self.h <= h_coarse, (
            f'h_coarse must be >= h (h={self.h}, h_coarse={h_coarse})'
        )

        # discretization step ratio
        k = int(h_coarse / self.h)

        self.psi = self.psi[::k]
        self.value_function = self.value_function[::k]
        self.u_opt = self.u_opt[::k]

        if self.sde.is_overdamped_langevin:
            self.V = self.V[::k]
            self.perturbed_potential = self.perturbed_potential[::k]
            self.dV = self.dV[::k]
            self.perturbed_drift = self.perturbed_drift[::k]

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
        return self.u_opt[idx, 0] if hasattr(self, 'u_opt') else None

    def get_perturbed_potential_and_drift(self) -> None:
        """Compute potentials, gradients, and perturbed drift fields."""

        # flatten domain_h
        x = np.expand_dims(self.sde.domain_h, axis=1)

        # diffusion term
        sigma = self.sde.diffusion

        # potential, bias potential and tilted potential
        self.V = np.squeeze(self.sde.potential(x))
        self.bias_potential = (sigma**2) * self.value_function
        self.perturbed_potential = self.V + self.bias_potential

        # gradient and tilted drift
        self.dV = self.sde.gradient(x)
        self.perturbed_drift = - self.dV + sigma * self.u_opt


    def write_report(self, x: float) -> None:
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
        u_opt_max = self.u_opt[idx_u_max, 0]
        print('argmax_x u_opt(x): {:2.3f}'.format(x_u_max))
        print('max_x u_opt(x): {:2.3f}'.format(u_opt_max))

        # computational time
        h, m, s = get_time_in_hms(self.ct)
        print('\nComputational time: {:d}:{:02d}:{:02.2f}\n'.format(h, m, s))

    def plot_1d_psi(
        self,
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plot the estimated solution psi(x).

        Args:
            xlim: Optional x-axis limits.
            ylim: Optional y-axis limits.

        Returns:
            Matplotlib figure and axes.
        """
        fig, ax = plt.subplots()
        ax.set_title(r'Estimation of $\Psi(x)$')
        ax.set_xlabel('x')
        ax.set_xlim(xlim) if xlim is not None else ax.set_xlim(self.sde.domain)
        if ylim is not None: ax.set_ylim(ylim)
        ax.plot(self.sde.domain_h, self.psi, lw=2.5)
        return fig, ax

    def plot_1d_value_function(
        self,
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plot the estimated value function phi(x).

        Args:
            xlim: Optional x-axis limits.
            ylim: Optional y-axis limits.

        Returns:
            Matplotlib figure and axes.
        """
        fig, ax = plt.subplots()
        ax.set_title(r'Estimation of $\Phi(x)$')
        ax.set_xlabel('x')
        ax.set_xlim(xlim) if xlim is not None else ax.set_xlim(self.sde.domain)
        if ylim is not None: ax.set_ylim(ylim)
        ax.plot(self.sde.domain_h, self.value_function, lw=2.5)
        return fig, ax

    def plot_1d_perturbed_potential(
        self,
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plot the original and perturbed potentials.

        Args:
            xlim: Optional x-axis limits.
            ylim: Optional y-axis limits.

        Returns:
            Matplotlib figure and axes.
        """
        fig, ax = plt.subplots()
        ax.set_title(r'Perturbed potential $(U_{pot} + U_{bias})(x)$')
        ax.set_xlabel('x')
        ax.set_xlim(xlim) if xlim is not None else ax.set_xlim(self.sde.domain)
        if ylim is not None: ax.set_ylim(ylim)
        ax.plot(self.sde.domain_h, self.V, lw=2.5)
        ax.plot(self.sde.domain_h, self.perturbed_potential, lw=2.5)
        return fig, ax

    def plot_1d_control(
        self,
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plot the optimal control u*(x).

        Args:
            xlim: Optional x-axis limits.
            ylim: Optional y-axis limits.

        Returns:
            Matplotlib figure and axes.
        """
        fig, ax = plt.subplots()
        ax.set_title(r'Optimal control $u^*(x)$')
        ax.set_xlabel('x')
        ax.set_xlim(xlim) if xlim is not None else ax.set_xlim(self.sde.domain)
        if ylim is not None: ax.set_ylim(ylim)
        ax.plot(self.sde.domain_h, self.u_opt, lw=2.5)
        return fig, ax

    def plot_1d_perturbed_drift(
        self,
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plot the perturbed drift field.

        Args:
            xlim: Optional x-axis limits.
            ylim: Optional y-axis limits.

        Returns:
            Matplotlib figure and axes.
        """
        fig, ax = plt.subplots()
        ax.set_title(r'Perturbed drift $\nabla(U_{pot} + U_{bias})(x)$')
        ax.set_xlabel('x')
        ax.set_xlim(xlim) if xlim is not None else ax.set_xlim(self.sde.domain)
        if ylim is not None: ax.set_ylim(ylim)
        self.get_perturbed_potential_and_drift()
        ax.plot(self.sde.domain_h, self.perturbed_drift, lw=2.5)
        return fig, ax

    def plot_1d_mfht(
        self,
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plot the mean first hitting time estimate.

        Args:
            xlim: Optional x-axis limits.
            ylim: Optional y-axis limits.

        Returns:
            Matplotlib figure and axes.
        """
        fig, ax = plt.subplots()
        ax.set_title(r'Estimation of $\mathbb{E}^x[\tau]$')
        ax.set_xlabel('x')
        ax.set_xlim(xlim) if xlim is not None else ax.set_xlim(self.sde.domain)
        if ylim is not None: ax.set_ylim(ylim)
        ax.plot(self.sde.domain_h, self.mfht, lw=2.5)
        return fig, ax

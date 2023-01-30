import functools

import matplotlib.pyplot as plt
import numpy as np

from sde_hjb_solver.functions import *

class ControlledSDE2D(object):
    '''
    '''

    def __init__(self, domain):

        # dimension
        self.d = 2

        # domain bounds
        self.domain = domain

    def discretize_domain_2d(self, h):
        '''
        '''

        # discretization step
        self.h = h

        # domain bounds
        x_lb, x_ub = self.domain[0, 0], self.domain[0, 1]
        y_lb, y_ub = self.domain[1, 0], self.domain[1, 1]

        # discretized domain
        self.domain_h = np.mgrid[
            slice(x_lb, x_ub + h, h),
            slice(y_lb, y_ub + h, h),
        ]
        self.domain_h = np.moveaxis(self.domain_h, 0, -1)

        # save number of indices per axis
        self.Nx = self.domain_h.shape[:-1]

        # save number of flattened indices
        self.Nh = self.Nx[0] * self.Nx[1]

        # get flat domain
        x = self.domain_h.reshape(self.Nh, self.d)

        # get node indices corresponding to boundary
        is_in_boundary_x = ((x[:, 0] == x_lb) | (x[:, 0] == x_ub))
        is_in_boundary_y = ((x[:, 1] == y_lb) | (x[:, 1] == y_ub))

        self.idx_boundary_x = np.where(is_in_boundary_x == True)[0]
        self.idx_boundary_y = np.where(is_in_boundary_y == True)[0]
        self.idx_boundary = np.where(
            (is_in_boundary_x == True) |
            (is_in_boundary_y == True)
        )[0]
        self.idx_corner = np.where(
            (is_in_boundary_x == True) &
            (is_in_boundary_y == True)
        )[0]

        # get node indices corresponding to the target set
        self.get_idx_target_set()

    def set_stopping_time_setting(self, lam=1.):
        '''
        '''

        # running and final costs
        self.lam = lam
        self.f = functools.partial(constant, a=lam)
        self.g = functools.partial(constant, a=0.)

        # target set indices
        self.get_idx_target_set = self.get_idx_target_set_stopping_time

    """
    def compute_mfht(self):
        ''' estimates the expected first hitting time by finite differences of the quantity
            of interest psi(x)
        '''
        from sde_hjb_solver.hjb_solver_1d import SolverHJB1D
        from copy import copy

        delta = 0.001

        sde_plus = copy(self)
        sde_plus.set_stopping_time_setting(lam=delta)
        sol_plus = SolverHJB1D(sde_plus, self.h)
        sol_plus.solve_bvp()

        sde_minus = copy(self)
        sde_minus.set_stopping_time_setting(lam=-delta)
        sol_minus = SolverHJB1D(sde_minus, self.h)
        sol_minus.solve_bvp()

        self.mfht = - (sol_plus.psi - sol_minus.psi) / (2 * delta)
    """

    def set_committor_setting(self, epsilon=1e-10):
        '''
        '''

        # running and final costs
        self.f = lambda x: 0
        self.epsilon = epsilon
        self.g = lambda x: np.where(
            (x >= self.target_set_b[:, 0]) &
            (x <= self.target_set_b[:, 1]),
            0,
            -np.log(self.epsilon),
        )[0]

        # target set indices
        self.get_idx_target_set = self.get_idx_target_set_committor


    def get_idx_target_set(self):
        raise NameError('Method not defined in subclass')

    def get_idx_target_set_stopping_time(self):
        '''
        '''
        # flatten domain_h
        x = self.domain_h.reshape(self.Nh, self.d)

        # boolean array telling us if x is in the target set
        is_in_target_set = (
            (x >= self.target_set[:, 0]) &
            (x <= self.target_set[:, 1])
        ).all(axis=1).reshape(self.Nh, 1)

        # get index
        self.idx_ts = np.where(is_in_target_set == True)[0]


    def get_idx_target_set_committor(self):
        '''
        '''

        # flatten domain_h
        x = self.domain_h.reshape(self.Nh, self.d)

        # boolean array telling us if x is in the target set a
        self.is_in_target_set_a = (
            (x >= self.target_set_a[:, 0]) &
            (x <= self.target_set_a[:, 1])
        ).all(axis=1).reshape(self.Nh, 1)

        # boolean array telling us if x is in the target set b
        self.is_in_target_set_b = (
            (x >= self.target_set_b[:, 0]) &
            (x <= self.target_set_b[:, 1])
        ).all(axis=1).reshape(self.Nh, 1)

        # indices of domain_h in tha target set A
        self.idx_ts_a = np.where(self.is_in_target_set_a == True)[0]
        self.idx_ts_b = np.where(self.is_in_target_set_b == True)[0]
        self.idx_ts = np.where(
            (self.is_in_target_set_a == True) | (self.is_in_target_set_b == True)
        )[0]

    def get_index(self, x):
        '''
        '''
        x = np.asarray(x)
        is_scalar = False

        if x.ndim == 0:
            is_scalar = True
            x = x[np.newaxis]

        if x.ndim != 1:
            raise ValueError('x array dimension must be one')

        idx = np.floor((
            np.clip(x, self.domain[0], self.domain[1] - 2 * self.h) + self.domain[1]
        ) / self.h).astype(int)

        if is_scalar:
            return np.squeeze(idx)
        elif x.shape[0] == 1:
            return idx
        else:
            return tuple(idx)

    def plot_2d_potential(self, ylim=None):
        fig, ax = plt.subplots()
        ax.set_title(r'Potential $V(x)$')
        ax.set_xlabel(r'$x_1$')
        ax.set_ylabel(r'$x_2$')
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)

        # flat domain
        x = self.domain_h.reshape(self.Nh, self.d)

        # contour f
        cs = ax.contourf(
            self.domain_h[:, :, 0],
            self.domain_h[:, :, 1],
            self.potential(x).reshape(self.Nx),
            extend='both',
        )

        # colorbar
        cbar = fig.colorbar(cs)

        plt.show()

    def plot_mfht(self, ylim=None):
        fig, ax = plt.subplots()
        ax.set_title(r'Estimation of $E[fht | X_0 = x]$')
        ax.set_xlabel('x')
        ax.set_xlim(self.domain)
        if ylim is not None:
            ax.set_ylim(ylim)
        x = self.domain_h
        y = self.mfht
        ax.plot(x, y)
        plt.show()


class OverdampedLangevinSDE2D(ControlledSDE2D):
    '''
    '''

    def __init__(self, beta=1., domain=None):
        super().__init__(domain=domain)

        # inverse temperature
        self.beta = beta

        # diffusion
        self.diffusion = np.sqrt(2 / self.beta)

class DoubleWellStoppingTime2D(OverdampedLangevinSDE2D):
    '''
    '''

    def __init__(self, beta=1., alpha=np.array([1., 1.]), lam=1.0, domain=None, target_set=None):
        super().__init__(beta=beta, domain=domain)

        # potential
        self.alpha = alpha
        self.potential = functools.partial(double_well, alpha=self.alpha)
        self.gradient = functools.partial(double_well_gradient, alpha=self.alpha)

        # drift term
        self.drift = lambda x: - self.gradient(x)

        # domain
        if self.domain is not None:
            self.domain = domain
        else:
            self.domain = np.full((self.d, 2), [-2, 2])

        # target set
        if target_set is not None:
            self.target_set = target_set
        else:
            self.target_set = np.full((self.d, 2), [1, 2])

        # stopping time setting
        self.set_stopping_time_setting(lam=lam)


class DoubleWellCommittor2D(OverdampedLangevinSDE2D):
    '''
    '''

    def __init__(self, beta=1., alpha=np.array([1., 1.]), epsilon=1e-10,
                 domain=None, target_set_a=None, target_set_b=None):
        super().__init__(beta=beta, domain=domain)

        # potential
        self.alpha = alpha
        self.potential = functools.partial(double_well, alpha=self.alpha)
        self.gradient = functools.partial(double_well_gradient, alpha=self.alpha)

        # drift term
        self.drift = lambda x: - self.gradient(x)

        # domain
        if self.domain is not None:
            self.domain = domain
        else:
            self.domain = np.full((self.d, 2), [-2, 2])

        # target sets
        if target_set_a is not None:
            self.target_set_a = target_set_a
        else:
            self.target_set_a = np.full((self.d, 2), [-2, -1])

        if target_set_b is not None:
            self.target_set_b = target_set_b
        else:
            self.target_set_b = np.full((self.d, 2), [1, 2])

        # committor setting
        self.set_committor_setting(epsilon)

"""
class BrownianMotionCommittor2D(ControlledSDE2D):
    '''
    '''

    def __init__(self, epsilon=1e-10):
        super().__init__()

        # drift and diffusion terms
        self.drift = lambda x: 0
        self.diffusion = 1.

        # domain
        self.domain = (-2, 2)

        # target set
        self.target_set_a = (-2, -1)
        self.target_set_b = (1, 2)

        # committor setting
        self.set_committor_setting(epsilon)
"""

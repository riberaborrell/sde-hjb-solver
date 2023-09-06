import functools

import matplotlib.pyplot as plt
import numpy as np

from sde_hjb_solver.functions import *

class ControlledSDE1D(object):
    '''
    '''

    def __init__(self, domain):

        # dimension
        self.d = 1

        # domain bounds
        self.domain = domain

    def discretize_domain_1d(self, h):
        '''
        '''

        # discretization step
        self.h = h

        # domain bounds
        lb, ub = self.domain[0], self.domain[1]

        # discretized domain
        self.domain_h = np.around(np.arange(lb, ub + h, h), decimals=3)

        # number of indices per axis
        self.Nx = self.domain_h.shape[0]

        # number of nodes
        self.Nh = self.domain_h.shape[0]

        # get node indices corresponding to the target set
        self.get_target_set_idx()

    def set_stopping_time_setting(self, lam=1.):
        '''
        '''

        # running and final costs
        self.lam = lam
        self.f = functools.partial(constant, a=lam)
        self.g = functools.partial(constant, a=0.)

        # target set indices
        self.get_target_set_idx = self.get_target_set_idx_stopping_time

    def compute_mfht(self):
        ''' estimates the expected first hitting time by finite differences of the quantity
            of interest psi(x)
        '''
        from sde_hjb_solver.hjb_solver_1d_st import SolverHJB1D
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

    def set_committor_setting(self, epsilon=1e-10):
        '''
        '''

        # running and final costs
        self.f = lambda x: 0
        self.epsilon = epsilon
        self.g = lambda x: np.where(
            (x >= self.target_set_b[0]) & (x <= self.target_set_b[1]),
            0,
            -np.log(self.epsilon),
        )

        # target set indices
        self.get_target_set_idx = self.get_target_set_idx_committor

    def get_target_set_idx(self):
        raise NameError('Method not defined in subclass')

    def get_target_set_idx_stopping_time(self):
        '''
        '''
        # target set lower and upper bound
        target_set_lb, target_set_ub = self.target_set[0], self.target_set[1]


        # indices of the discretized domain corresponding to the target set
        is_in_target_set = (self.domain_h >= target_set_lb) & (self.domain_h <= target_set_ub)
        self.ts_idx = np.where(is_in_target_set)[0]

    def get_target_set_idx_committor(self):
        '''
        '''
        # indices of domain_h in tha target set A
        self.ts_a_idx = np.where(
            (self.domain_h >= self.target_set_a[0]) & (self.domain_h <= self.target_set_a[1])
        )[0]

        # indices of domain_h in tha target set B
        self.ts_b_idx = np.where(
            (self.domain_h >= self.target_set_b[0]) & (self.domain_h <= self.target_set_b[1])
        )[0]

        # indices of the discretized domain corresponding to the target set
        self.ts_idx = np.where(
            ((self.domain_h >= self.target_set_a[0]) & (self.domain_h <= self.target_set_a[1])) |
            ((self.domain_h >= self.target_set_b[0]) & (self.domain_h <= self.target_set_b[1]))
        )[0]

    def get_idx(self, x):
        ''' get index of the grid point which approximates x
        '''
        # array convertion
        x = np.asarray(x)

        # scalar input
        if x.ndim == 0:
            is_scalar = True
            x = x[np.newaxis]
        else:
            is_scalar = False

        if x.ndim != 1:
            raise ValueError('x array dimension must be one')

        idx = self.get_idx_truncate(x)
        #idx = self.get_idx_min(x)

        if is_scalar:
            return np.squeeze(idx)
        elif x.shape[0] == 1:
            return idx
        else:
            return tuple(idx)

    def get_idx_truncate(self, x):
        x = np.clip(x, self.domain[0], self.domain[1])
        idx = np.floor((x - self.domain[0]) / self.h).astype(int)
        return idx

    def get_idx_min(self, x):
        return np.argmin(np.abs(self.domain_h - x), axis=1)

    def plot_1d_potential(self, ylim=None):
        fig, ax = plt.subplots()
        ax.set_title(r'Potential $V(x)$')
        ax.set_xlabel('x')
        ax.set_xlim(self.domain)
        if ylim is not None:
            ax.set_ylim(ylim)
        x = np.expand_dims(self.domain_h, axis=1)
        y = np.squeeze(self.potential(x))
        ax.plot(x, y)
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


class OverdampedLangevinSDE1D(ControlledSDE1D):
    '''
    '''

    def __init__(self, beta=1., domain=None):
        super().__init__(domain=domain)

        # overdamped langevin flag
        self.is_overdamped_langevin = True

        # inverse temperature
        self.beta = beta

        # diffusion
        self.diffusion = np.sqrt(2 / self.beta)

class DoubleWellStoppingTime1D(OverdampedLangevinSDE1D):
    '''
    '''

    def __init__(self, beta=1., alpha=1., lam=1.0, domain=None, target_set=None):
        super().__init__(beta=beta, domain=domain)

        # log name
        self.name = 'doublewell-1d-st__beta{:.1f}_alpha{:.1f}'.format(beta, alpha)

        # potential
        self.alpha = alpha
        self.potential = functools.partial(double_well, alpha=self.alpha)
        self.gradient = functools.partial(double_well_gradient, alpha=self.alpha)

        # drift term
        self.drift = lambda x: - self.gradient(x)

        # domain
        if self.domain is None:
            self.domain = (-2, 2)

        # target set
        if target_set is not None:
            self.target_set = target_set
        else:
            self.target_set = (1, 2)

        # stopping time setting
        self.set_stopping_time_setting(lam=lam)


class DoubleWellCommittor1D(OverdampedLangevinSDE1D):
    '''
    '''

    def __init__(self, beta=1., alpha=1., epsilon=1e-10,
                 domain=None, target_set_a=None, target_set_b=None):
        super().__init__(beta=beta, domain=domain)

        # log name
        self.name = 'doublewell-1d-committor__beta{:.1f}_alpha{:.1f}'.format(beta, alpha)

        # potential
        self.alpha = alpha
        self.potential = functools.partial(double_well, alpha=self.alpha)
        self.gradient = functools.partial(double_well_gradient, alpha=self.alpha)

        # drift term
        self.drift = lambda x: - self.gradient(x)

        # domain
        if self.domain is None:
            self.domain = (-2, 2)

        # target set
        if target_set_a is not None:
            self.target_set_a = target_set_a
        else:
            self.target_set_a = (-2, -1)

        if target_set_b is not None:
            self.target_set_b = target_set_b
        else:
            self.target_set_b = (1, 2)

        # committor setting
        self.set_committor_setting(epsilon)

class FiveWellCommittor1D(OverdampedLangevinSDE1D):
    '''
    '''

    def __init__(self, beta=1., epsilon=1e-10, domain=None):
        super().__init__(beta=beta, domain=domain)

        # log name
        self.name = 'fivewell-1d-committor__beta{:.1f}'.format(beta)


        # potential
        self.potential = lambda x: + (0.5 * x**6 - 15 * x**4 + 119 * x**2 + 28*x + 50) / 200 \
                                   - 0.6 * np.exp(-0.5 *(x + 2)**2 / (0.2)**2) \
                                   - 0.7 * np.exp(-0.5*(x-1.8)**2/(0.2)**2)

        # drift term
        self.gradient = lambda x: + (3 * x**5 - 60 * x**3 + 238 * x + 28) / 200 \
                               - 0.6 * (x + 2) * np.exp(-0.5 *(x + 2)**2 / (0.2)**2) / (0.2)**2 \
                               - 0.7 * (x - 1.8) * np.exp(-0.5*(x - 1.8)**2/(0.2)**2) / (0.2)**2

        # drift term
        self.drift = lambda x: - self.gradient(x)

        # domain
        if self.domain is None:
            self.domain = (-5, 5)

        # target set
        x1 = -3.9
        x2 = -0.1
        x3 = 3.9
        #self.target_set_a = (x3 - 0.1, x3 + 0.1)
        self.target_set_a = (x2 - 0.1, x2 + 0.1)
        self.target_set_b = (x1 - 0.1, x1 + 0.1)

        # committor setting
        self.set_committor_setting(epsilon)


class SkewDoubleWellStoppingTime1D(OverdampedLangevinSDE1D):
    ''' We aim to compute the MFHT of a process following overdamped langevin
        dynamics with skew double well potential (see Hartmann2012).
    '''

    def __init__(self, beta=1., lam=1., domain=None, target_set=None):
        super().__init__(beta=beta, domain=domain)

        # log name
        self.name = 'skewdoublewell-1d-st__beta{:.1f}'.format(beta)

        # potential
        self.potential = skew_double_well_1d
        self.gradient = skew_double_well_gradient_1d

        # drift term
        self.drift = lambda x: - self.gradient(x)

        # domain
        if self.domain is None:
            self.domain = (-2, 2)

        # target set
        if target_set is not None:
            self.target_set = target_set
        else:
            self.target_set = (-1.1, -1)

        # stopping time setting
        self.set_stopping_time_setting(lam=lam)


class BrownianMotionCommittor1D(ControlledSDE1D):
    '''
    '''

    def __init__(self, epsilon=1e-10, domain=None, target_set_a=None, target_set_b=None):
        super().__init__(domain=domain)

        # log name
        self.name = 'brownian-1d-committor'.format()

        # drift and diffusion terms
        self.drift = lambda x: 0
        self.diffusion = 1.

        # domain
        if self.domain is None:
            self.domain = (-2, 2)

        # target set
        if target_set_a is not None:
            self.target_set_a = target_set_a
        else:
            self.target_set_a = (-2, -1)

        if target_set_b is not None:
            self.target_set_b = target_set_b
        else:
            self.target_set_b = (1, 2)

        # committor setting
        self.set_committor_setting(epsilon)

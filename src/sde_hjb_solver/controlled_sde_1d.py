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

        # problem types flags
        self.is_mgf = False
        self.is_committor = False
        self.overdamped_langevin = False

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

    def set_mgf_setting(self, lam=1.):
        ''' Set moment generating function of the first hitting time setting
        '''
        # set mgf problem flag
        self.is_mgf = True

        # set in target set condition function
        self.is_in_target_set = lambda x: (x >= self.target_set[0]) & (x <= self.target_set[1])

        # running and final costs
        self.lam = lam
        self.f = functools.partial(constant, a=lam)
        self.g = functools.partial(constant, a=0.)

        # target set indices
        self.get_target_set_idx = self.get_target_set_idx_mgf


    def set_committor_setting(self, epsilon=1e-10):
        ''' Set committor probability setting
        '''
        # set committor problem flag
        self.is_committor = True

        # set in target set condition functions
        self.is_in_target_set_a = lambda x: (x >= self.target_set_a[0]) & (x <= self.target_set_a[1])
        self.is_in_target_set_b = lambda x: (x >= self.target_set_b[0]) & (x <= self.target_set_b[1])

        # running and final costs
        self.epsilon = epsilon
        self.f = lambda x: 0
        self.g = lambda x: np.where(
            self.is_in_target_set_b(x),
            -np.log(1+epsilon),
            -np.log(epsilon),
        )

        # target set indices
        self.get_target_set_idx = self.get_target_set_idx_committor

    def get_target_set_idx(self):
        raise NameError('Method not defined in subclass')

    def get_target_set_idx_mgf(self):
        '''
        '''
        # indices of the discretized domain corresponding to the target set
        x = self.domain_h
        self.ts_idx = np.where(self.is_in_target_set(x))[0]

    def get_target_set_idx_committor(self):
        '''
        '''
        # indices of the discretized domain corresponding to the target sets
        x = self.domain_h
        self.ts_a_idx = np.where(self.is_in_target_set_a(x) == True)[0]
        self.ts_b_idx = np.where(self.is_in_target_set_b(x) == True)[0]
        self.ts_idx = np.where(
            (self.is_in_target_set_a(x) == True) | (self.is_in_target_set_b(x) == True)
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

    def compute_mfht(self, delta=0.001):
        ''' estimates the expected first hitting time by finite differences of the quantity
            of interest psi(x)
        '''
        from sde_hjb_solver.hjb_solver_1d_st import SolverHJB1D
        from copy import copy

        sde_plus = copy(self)
        sde_plus.set_mgf_setting(lam=delta)
        sol_plus = SolverHJB1D(sde_plus, self.h)
        sol_plus.solve_bvp()

        sde_minus = copy(self)
        sde_minus.set_mgf_setting(lam=-delta)
        sol_minus = SolverHJB1D(sde_minus, self.h)
        sol_minus.solve_bvp()

        return - (sol_plus.psi - sol_minus.psi) / (2 * delta)

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

    def psi_ana(self, x):
        a = self.target_set_a[1]
        b = self.target_set_b[0]

        return np.where(
            x < a,
            0,
            np.where(x <= b, (x - a) / (b - a), 1),
        )

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

    def plot_1d_potential(self, xlim=None, ylim=None):
        fig, ax = plt.subplots()
        ax.set_title(r'Potential $V(x)$')
        ax.set_xlabel('x')
        ax.set_xlim(self.domain)
        ax.set_xlim(xlim) if xlim is not None else ax.set_xlim(self.domain)
        if ylim is not None:
            ax.set_ylim(ylim)
        x = np.expand_dims(self.domain_h, axis=1)
        y = np.squeeze(self.potential(x))
        ax.plot(x, y, lw=2.5)
        plt.show()


class DoubleWell1D(OverdampedLangevinSDE1D):
    ''' Overdamped langevin dynamics with double well potential.
    '''
    def __init__(self, beta=1., alpha=1., domain=None):
        super().__init__(beta=beta, domain=domain)

        # potential
        self.alpha = alpha
        self.potential = functools.partial(double_well, alpha=self.alpha)
        self.gradient = functools.partial(double_well_gradient, alpha=self.alpha)

        # drift term
        self.drift = lambda x: - self.gradient(x)

        # domain
        if self.domain is None:
            self.domain = (-2, 2)

class DoubleWellMGF1D(DoubleWell1D):

    def __init__(self, beta=1., alpha=1., lam=1.0, domain=None, target_set=None):
        super().__init__(beta=beta, alpha=alpha, domain=domain)

        # log name
        self.name = 'doublewell-1d-mgf__beta{:.1f}_alpha{:.1f}'.format(beta, alpha)

        # target set
        if target_set is not None:
            self.target_set = target_set
        else:
            self.target_set = (1, 2)

        # first hitting time setting
        self.set_mgf_setting(lam=lam)


class DoubleWellCommittor1D(DoubleWell1D):

    def __init__(self, beta=1., alpha=1., epsilon=1e-10,
                 domain=None, target_set_a=None, target_set_b=None):
        super().__init__(beta=beta, alpha=alpha, domain=domain)

        # log name
        self.name = 'doublewell-1d-committor__beta{:.1f}_alpha{:.1f}'.format(beta, alpha)

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

class SkewDoubleWell1D(OverdampedLangevinSDE1D):
    ''' Overdamped langevin dynamics with skew double well potential.
    '''

    def __init__(self, beta=1., domain=None):
        super().__init__(beta=beta, domain=domain)

        # potential
        self.potential = skew_double_well_1d
        self.gradient = skew_double_well_gradient_1d

        # drift term
        self.drift = lambda x: - self.gradient(x)

        # domain
        if self.domain is None:
            self.domain = (-2, 2)

class SkewDoubleWellMGF1D(SkewDoubleWell1D):
    ''' Moment generating function setting. We aim to compute the MFHT (see Hartmann2012).
    '''

    def __init__(self, beta=1., lam=1., domain=None, target_set=None):
        super().__init__(beta=beta, domain=domain)

        # log name
        self.name = 'skewdoublewell-1d-mgf__beta{:.1f}'.format(beta)

        # target set
        if target_set is not None:
            self.target_set = target_set
        else:
            self.target_set = (-1.1, -1)

        # first hitting time setting
        self.set_mgf_setting(lam=lam)

class TripleWell1D(OverdampedLangevinSDE1D):
    ''' Overdamped langevin dynamics with asymmetric triple well potential.
    '''

    def __init__(self, beta=1., domain=None):
        super().__init__(beta=beta, domain=domain)

        # potential
        self.potential = triple_well_1d
        self.gradient = triple_well_gradient_1d

        # drift term
        self.drift = lambda x: - self.gradient(x)

        # domain
        if self.domain is None:
            self.domain = (-5, 5)

        # local minima (using wolfram alpha)
        self.m1 = -3.84778 # global minimum
        self.m2 = -0.118062 # second lowest minimum
        self.m3 = 3.77699 # highest minimum

class TripleWellMGF1D(TripleWell1D):
    '''
    '''

    def __init__(self, beta=1., lam=1.0, domain=None):
        super().__init__(beta=beta, domain=domain)

        # log name
        self.name = 'triplewell-1d-mgf__beta{:.1f}'.format(beta)

        # target set
        self.target_set = (self.m1 - 0.1, self.m1 + 0.1)

        # first hitting time setting
        self.set_mgf_setting(lam=lam)


class TripleWellCommittor1D(TripleWell1D):
    '''
    '''

    def __init__(self, beta=1., epsilon=1e-10, domain=None, ts_a='m2'):
        super().__init__(beta=beta, domain=domain)

        # log name
        self.name = 'triplewell-1d-committor__beta{:.1f}'.format(beta)

        # target set
        if ts_a == 'm2':
            ts_a_c = self.m2
        elif ts_a == 'm3':
            ts_a_c = self.m3

        self.target_set_a = (ts_a_c - 0.1, ts_a_c + 0.1)
        self.target_set_b = (self.m1 - 0.1, self.m1 + 0.1)

        # committor setting
        self.set_committor_setting(epsilon)

class FiveWell1D(OverdampedLangevinSDE1D):
    ''' Overdamped langevin dynamics with five well potential.
    '''

    def __init__(self, beta=1., domain=None):
        super().__init__(beta=beta, domain=domain)

        # potential
        self.potential = five_well_1d
        self.gradient = five_well_gradient_1d

        # drift term
        self.drift = lambda x: - self.gradient(x)

        # domain
        if self.domain is None:
            self.domain = (-5, 5)

        # local minima (using wolfram alpha)
        self.m1 = -3.84778 # global minimum
        self.m4 = -1.97665
        self.m2 = -0.118062 # second lowest minimum
        self.m5 = 1.74915
        self.m3 = 3.77699 # highest minimum

class FiveWellMGF1D(FiveWell1D):
    '''
    '''

    def __init__(self, beta=1., lam=1.0, domain=None):
        super().__init__(beta=beta, domain=domain)

        # log name
        self.name = 'fivewell-1d-mgf__beta{:.1f}'.format(beta)

        # target set
        self.target_set = (self.m1 - 0.1, self.m1 + 0.1)

        # first hitting time setting
        self.set_mgf_setting(lam=lam)


class FiveWellCommittor1D(FiveWell1D):
    '''
    '''

    def __init__(self, beta=1., epsilon=1e-10, domain=None, ts_a='m2'):
        super().__init__(beta=beta, domain=domain)

        # log name
        self.name = 'fivewell-1d-committor__beta{:.1f}'.format(beta)

        # target set
        if ts_a == 'm2':
            ts_a_c = self.m2
        elif ts_a == 'm3':
            ts_a_c = self.m3

        self.target_set_a = (ts_a_c - 0.1, ts_a_c + 0.1)
        self.target_set_b = (self.m1 - 0.1, self.m1 + 0.1)

        # committor setting
        self.set_committor_setting(epsilon)

import functools

import matplotlib.pyplot as plt
import numpy as np

from sde_hjb_solver.functions import *
from sde_hjb_solver.controlled_sde import ControlledSDE

class ControlledSDE1D(ControlledSDE):
    '''
    '''

    def __init__(self, **kwargs):

        # dimension
        kwargs.update(d=1)

        super().__init__(**kwargs)

    def discretize_domain_1d(self, h):
        '''
        '''

        # discretization step
        self.h = h

        # domain bounds
        lb, ub = self.domain[0], self.domain[1]

        # discretized domain
        self.domain_h = np.around(np.arange(lb, ub + h, h), decimals=3)
        #self.domain_h = (self.domain_h + h/2)[:-1]

        # number of indices per axis
        self.Nx = self.domain_h.shape[0]

        # number of nodes
        self.Nh = self.domain_h.shape[0]

        # get node indices corresponding to the target set
        self.get_target_set_idx()

    def get_index_vectorized(self, x):
        '''
        '''
        assert x.ndim == 2, ''
        assert x.shape[1] == self.d, ''

        # domain bounds
        lb, ub = self.domain[0], self.domain[1]

        # clip x to domain
        x = np.clip(x, lb, ub)

        # get index by truncation
        idx = np.floor((x - lb) / self.h).astype(int)
        idx = tuple([idx[:, i] for i in range(self.d)])
        return idx


    def set_is_target_set_mgf(self):
        '''set is in target set condition function'''
        self.is_target_set = lambda x: (x >= self.target_set[0]) & \
                                          (x <= self.target_set[1])


    def set_is_target_set_committor(self):
        '''set is in target set condition functions'''
        self.is_target_set_a = lambda x: (x >= self.target_set_a[0]) & (x <= self.target_set_a[1])
        self.is_target_set_b = lambda x: (x >= self.target_set_b[0]) & (x <= self.target_set_b[1])

    def get_target_set_idx_mgf(self):
        '''get indices of the discretized domain corresponding to the target set
        '''
        x = self.domain_h
        self.ts_idx = np.where(self.is_target_set(x))[0]

    def get_target_set_idx_committor(self):
        '''get indices of the discretized domain corresponding to the target sets A and B
        '''
        x = self.domain_h
        self.ts_a_idx = np.where(self.is_target_set_a(x))[0]
        self.ts_b_idx = np.where(self.is_target_set_b(x))[0]
        self.ts_idx = np.where(self.is_target_set_a(x) | self.is_target_set_b(x))[0]

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

class BrownianMotion1D(ControlledSDE1D):
    '''
    '''

    def __init__(self, beta: float = 1, **kwargs):
        super().__init__(**kwargs)

        # log name
        self.name = 'brownian-1d-'

        # overdamped langevin flag
        self.is_overdamped_langevin = False

        # drift and diffusion terms
        self.drift = lambda x: 0
        self.beta = beta
        self.diffusion = np.sqrt(2 / self.beta)

        # domain
        if self.domain is None:
            self.domain = (-1.5, 1.5)

        # parameters string
        self.params_str = 'sigma{:.1f}'.format(self.diffusion)

class BrownianMotionMgf1D(BrownianMotion1D):
    '''
    '''

    def __init__(self, lam=1.0, target_set_r=1., target_set_c=0., **kwargs):
        super().__init__(**kwargs)

        # log name
        self.name += 'mgf'

        # target set center and radius
        self.target_set_c = target_set_c
        self.target_set_r = target_set_r

        # first hitting time setting
        self.set_mgf_setting(lam=lam)

        # set in target set condition function
        self.is_target_set = lambda x: np.abs(x-self.target_set_c) >= self.target_set_r

    def psi_ana(self, x):
        c = self.target_set_c
        return np.where(
            self.is_target_set(x),
            1,
            np.cosh(np.sqrt(self.beta) * (x-c)) / np.cosh(np.sqrt(self.beta)),
        )

    def u_opt_ana(self, x):
        c = self.target_set_c
        return np.where(
            self.is_target_set(x),
            0,
            np.sqrt(2) * np.tanh(np.sqrt(self.beta) * (x-c)),
        )

    def mfht_ana(self, x):
        c, r = self.target_set_c, self.target_set_r
        return np.where(
            self.is_target_set(x),
            0,
            (r**2 - np.abs(x-c)**2)/(self.diffusion**2),
        )

class BrownianMotionCommittor1D(BrownianMotion1D):
    '''
    '''

    def __init__(self, epsilon=1e-10, target_set_a=(-2, -1), target_set_b=(1, 2), **kwargs):
        super().__init__(**kwargs)

        # log name
        self.name += 'committor'

        # target sets
        self.target_set_a = target_set_a
        self.target_set_b = target_set_b

        # committor setting
        self.set_committor_setting(epsilon)

        # set in target set condition function
        self.set_is_target_set_committor()

    def psi_ana(self, x):
        a = self.target_set_a[1]
        b = self.target_set_b[0]

        #TODO: generalize to arbitrary scaling factor sigma
        return np.where(
            x < a,
            0,
            np.where(x <= b, (x - a) / (b - a), 1),
        )

class OverdampedLangevinSDE1D(ControlledSDE1D):
    '''
    '''

    def __init__(self, beta: float = 1., **kwargs):
        super().__init__(**kwargs)

        # overdamped langevin flag
        self.is_overdamped_langevin = True

        # inverse temperature
        self.beta = beta

        # diffusion
        self.diffusion = np.sqrt(2 / self.beta)

    def plot_1d_potential(self, xlim=None, ylim=None):
        fig, ax = plt.subplots()
        ax.set_title(r'Potential $U_{pot}(x)$')
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
    def __init__(self, alpha=np.array([1.]), **kwargs):
        super().__init__(**kwargs)

        # check alpha
        assert alpha.size == 1, 'alpha must be an array of size 1'
        self.alpha = alpha[0]

        # log name
        self.name = 'doublewell-1d-'
        self.params_str = 'beta{:.1f}_alpha{:.1f}'.format(self.beta, self.alpha)

        # potential
        self.potential = functools.partial(double_well, alpha=self.alpha)
        self.gradient = functools.partial(double_well_gradient, alpha=self.alpha)

        # drift term
        self.drift = lambda x: - self.gradient(x)

        # domain
        if self.domain is None:
            self.domain = (-2, 2)

class DoubleWellMgf1D(DoubleWell1D):

    def __init__(self, lam=1.0, target_set=(1, 2), **kwargs):
        super().__init__(**kwargs)

        # log name
        self.name += 'mgf'

        # target set
        self.target_set = target_set

        # first hitting time setting
        self.set_mgf_setting(lam=lam)

        # set in target set condition function
        self.set_is_target_set_mgf()


class DoubleWellCommittor1D(DoubleWell1D):

    def __init__(self, epsilon=1e-10, target_set_a=(-2, -1), target_set_b=(1, 2), **kwargs):
        super().__init__(**kwargs)

        # log name
        self.name += 'committor'

        # target set
        self.target_set_a = target_set_a
        self.target_set_b = target_set_b

        # committor setting
        self.set_committor_setting(epsilon)

        # set in target set condition function
        self.set_is_target_set_committor()



class SkewDoubleWell1D(OverdampedLangevinSDE1D):
    ''' Overdamped langevin dynamics with skew double well potential.
    '''

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # log name
        self.name = 'skewdoublewell-1d-'
        self.params_str = 'beta{:.1f}'.format(self.beta)

        # potential
        self.potential = skew_double_well_1d
        self.gradient = skew_double_well_gradient_1d

        # drift term
        self.drift = lambda x: - self.gradient(x)

        # domain
        if self.domain is None:
            self.domain = (-2, 2)

class SkewDoubleWellMgf1D(SkewDoubleWell1D):
    ''' Moment generating function setting. We aim to compute the MFHT (see Hartmann2012).
    '''

    def __init__(self, lam=1.0, target_set=(-1.1, -1), **kwargs):
        super().__init__(**kwargs)

        # log name
        self.name += 'mgf'

        # target set
        self.target_set = target_set

        # first hitting time setting
        self.set_mgf_setting(lam=lam)

        # set in target set condition function
        self.set_is_target_set_mgf()

class TripleWell1D(OverdampedLangevinSDE1D):
    ''' Overdamped langevin dynamics with asymmetric triple well potential.
    '''

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # log name
        self.name = 'triplewell-1d-'
        self.params_str = 'beta{:.1f}'.format(self.beta)

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

class TripleWellMgf1D(TripleWell1D):
    '''
    '''

    def __init__(self, lam=1.0, **kwargs):
        super().__init__(**kwargs)

        # log name
        self.name += 'mgf'

        # target set
        self.target_set = (self.m1 - 0.1, self.m1 + 0.1)

        # first hitting time setting
        self.set_mgf_setting(lam=lam)

        # set in target set condition function
        self.set_is_target_set_mgf()

class TripleWellCommittor1D(TripleWell1D):
    '''
    '''

    def __init__(self, epsilon=1e-10, ts_a='m2', **kwargs):
        super().__init__(**kwargs)

        # log name
        self.name += 'committor'

        # target set
        if ts_a == 'm2':
            ts_a_c = self.m2
        elif ts_a == 'm3':
            ts_a_c = self.m3

        self.target_set_a = (ts_a_c - 0.1, ts_a_c + 0.1)
        self.target_set_b = (self.m1 - 0.1, self.m1 + 0.1)

        # committor setting
        self.set_committor_setting(epsilon)

        # set in target set condition function
        self.set_is_target_set_committor()

class RyckBell1D(OverdampedLangevinSDE1D):
    ''' Overdamped langevin dynamics with ryck bell potential.
    '''

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # log name
        self.name = 'ryck-bell-1d-'
        self.params_str = 'beta{:.1f}'.format(self.beta)

        # potential
        self.potential = lambda x: ryck_bell_1d(x - np.pi)
        self.gradient = lambda x: ryck_bell_gradient_1d(x - np.pi)

        # drift term
        self.drift = lambda x: - self.gradient(x)

        # domain
        if self.domain is None:
            #self.domain = (-np.pi, np.pi)
            self.domain = (0, 2 * np.pi)

        # local minima
        #self.m_trans = 0. # global minimum
        #self.m_gauche1 = 2 * np.pi / 3 # local minimum
        #self.m_gauche2 = - 2 * np.pi / 3 # local minimum
        self.m_trans = np.pi # global minimum
        self.m_gauche1 = np.pi / 3 # local minimum
        self.m_gauche2 = 5 * np.pi / 3 # local minimum

class RyckBellMgf1D(RyckBell1D):
    '''
    '''

    def __init__(self, lam=1.0, epsilon=20, **kwargs):
        super().__init__(**kwargs)

        # log name
        self.name += 'mgf'

        # target set
        self.epsilon = epsilon * np.pi / 180

        # first hitting time setting
        self.set_mgf_setting(lam=lam)

        # set in target set condition function
        self.is_target_set = lambda x: (np.abs(x - self.m_gauche1) <= self.epsilon) \
                                        | (np.abs(x - self.m_gauche2) <= self.epsilon)


class FiveWell1D(OverdampedLangevinSDE1D):
    ''' Overdamped langevin dynamics with five well potential.
    '''

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # log name
        self.name = 'fivewell-1d-'
        self.params_str = 'beta{:.1f}'.format(self.beta)

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

class FiveWellMgf1D(FiveWell1D):
    '''
    '''

    def __init__(self, lam=1.0, **kwargs):
        super().__init__(**kwargs)

        # log name
        self.name += 'mgf'

        # target set
        self.target_set = (self.m1 - 0.1, self.m1 + 0.1)

        # first hitting time setting
        self.set_mgf_setting(lam=lam)

        # set in target set condition function
        self.set_is_target_set_mgf()

class FiveWellCommittor1D(FiveWell1D):
    '''
    '''

    def __init__(self, epsilon=1e-10, ts_a='m2', **kwargs):
        super().__init__(**kwargs)

        # log name
        self.name += 'committor'

        # target set
        if ts_a == 'm2':
            ts_a_c = self.m2
        elif ts_a == 'm3':
            ts_a_c = self.m3

        self.target_set_a = (ts_a_c - 0.1, ts_a_c + 0.1)
        self.target_set_b = (self.m1 - 0.1, self.m1 + 0.1)

        # committor setting
        self.set_committor_setting(epsilon)

        # set in target set condition function
        self.set_is_target_set_committor()

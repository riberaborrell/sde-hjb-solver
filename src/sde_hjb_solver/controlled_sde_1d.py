import functools

import numpy as np

from sde_hjb_solver.functions import *

class ControlledSDE1D(object):
    '''
    '''

    def __init__(self):

        # dimension
        self.d = 1

    def discretize_domain_1d(self, h):
        '''
        '''

        # discretization step
        self.h = h

        # domain bounds
        lb, ub = self.domain[0], self.domain[1]

        # discretized domain
        self.domain_h = np.arange(lb, ub + h, h)

        # number of nodes
        self.Nh = self.domain_h.shape[0]

        # get node indices corresponding to the target set
        self.get_idx_target_set()

    def get_index(self, x):
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

    def get_idx_target_set(self):
        raise NameError('Method not defined in subclass')

class OverdampedLangevinSDE1D(ControlledSDE1D):
    '''
    '''

    def __init__(self, beta=1.):
        super().__init__()

        # inverse temperature
        self.beta = beta

        # diffusion
        self.diffusion = np.sqrt(2 / self.beta)

class DoubleWellStoppingTime1D(OverdampedLangevinSDE1D):
    '''
    '''

    def __init__(self, beta=1., alpha=1.):
        super().__init__(beta=beta)

        # potential
        self.alpha = alpha
        self.potential = functools.partial(double_well, alpha=self.alpha)
        self.gradient = functools.partial(double_well_gradient, alpha=self.alpha)

        # drift term
        self.drift = self.gradient

        # domain
        self.domain = (-2, 2)

        # target set
        self.target_set = (1, 2)

        # running and final costs
        self.f = functools.partial(constant, a=1.)
        self.g = functools.partial(constant, a=0.)

    def get_idx_target_set(self):

        # target set lower and upper bound
        target_set_lb, target_set_ub = self.target_set[0], self.target_set[1]

        # indices of the discretized domain corresponding to the target set
        self.idx_ts = np.where(
            (self.domain_h >= target_set_lb) & (self.domain_h <= target_set_ub)
        )[0]

class DoubleWellCommittor1D(OverdampedLangevinSDE1D):
    '''
    '''

    def __init__(self, beta=1., alpha=1., epsilon=1e-10):
        super().__init__(beta=beta)

        # potential
        self.alpha = alpha
        self.potential = functools.partial(double_well, alpha=self.alpha)
        self.gradient = functools.partial(double_well_gradient, alpha=self.alpha)
        #self.potential = lambda x: + (0.5 * x**6 - 15 * x**4 + 119 * x**2 + 28*x + 50) / 200 \
        #                           - 0.6 * np.exp(-0.5 *(x + 2)**2 / (0.2)**2) \
        #                           - 0.7 * np.exp(-0.5*(x-1.8)**2/(0.2)**2)

        # drift term
        #self.gradient = lambda x: + (3 * x**5 - 60 * x**3 + 238 * x + 28) / 200 \
        #                       - 0.6 * (x + 2) * np.exp(-0.5 *(x + 2)**2 / (0.2)**2) / (0.2)**2 \
        #                       - 0.7 * (x - 1.8) * np.exp(-0.5*(x - 1.8)**2/(0.2)**2) / (0.2)**2

        self.drift = self.gradient

        # domain
        self.domain = (-2, 2)
        #self.domain = (-5, 5)

        # target set
        self.target_set_a = (-2, -1)
        self.target_set_b = (1, 2)
        #self.target_set_a = (3.7, 5)
        #self.target_set_b = (-4.1, -3.7)

        # running and final costs
        self.f = lambda x: 0
        self.epsilon = epsilon
        self.g = lambda x: np.where(
            (x >= self.target_set_b[0]) & (x <= self.target_set_b[1]),
            0,
            -np.log(self.epsilon),
        )

    def get_idx_target_set(self):

        # indices of domain_h in tha target set A
        self.idx_ts_a = np.where(
            (self.domain_h >= self.target_set_a[0]) & (self.domain_h <= self.target_set_a[1])
        )[0]

        # indices of domain_h in tha target set B
        self.idx_ts_b = np.where(
            (self.domain_h >= self.target_set_b[0]) & (self.domain_h <= self.target_set_b[1])
        )[0]

        # indices of the discretized domain corresponding to the target set
        self.idx_ts = np.where(
            ((self.domain_h >= self.target_set_a[0]) & (self.domain_h <= self.target_set_a[1])) |
            ((self.domain_h >= self.target_set_b[0]) & (self.domain_h <= self.target_set_b[1]))
        )[0]

class SkewDoubleWellStoppingTime1D(OverdampedLangevinSDE1D):
    ''' We aim to compute the MFHT of a process following overdamped langevin
        dynamics with skew double well potential (see Hartmann2012).
    '''

    def __init__(self, beta=1., sigma=1.):
        super().__init__(beta=beta)

        # sigma
        self.sigma = sigma

        # potential
        self.potential = skew_double_well_1d
        self.gradient = skew_double_well_gradient_1d

        # drift term
        self.drift = self.gradient

        # domain
        self.domain = (-2, 2)

        # target set
        self.target_set = (-1.1, -1)

        # running and final costs
        self.f = lambda x: self.sigma * self.beta
        self.g = lambda x: 0

    def get_idx_target_set(self):

        # target set lower and upper bound
        target_set_lb, target_set_ub = self.target_set[0], self.target_set[1]

        # indices of the discretized domain corresponding to the target set
        self.idx_ts = np.where(
            (self.domain_h >= target_set_lb) & (self.domain_h <= target_set_ub)
        )[0]

    def compute_mfht(self):
        ''' computes the expected first hitting time by finite differences of the quantity
            of interest psi(x)
        '''
        #from hjb_solver_1d import SolverHJB1d
        from copy import copy

        def f(x, c):
            return c

        breakpoint()
        l = 0.001
        sde_plus = copy(self)
        sde_plus.sigma = l

        sol_plus = SolverHJB1d(self.sde, self.h)
        #f=functools.partial(f, c=l),
        #    g=self.g,
        #)
        sol_plus.discretize_domain()
        sol_plus.solve_bvp()
        sol_minus.discretize_domain()
        sol_minus.solve_bvp()

        self.exp_fht = - (sol_plus.Psi - sol_minus.Psi) / (self.beta * 2 * l)

class BrownianMotionCommittor1D(ControlledSDE1D):
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

        # running and final costs
        self.f = lambda x: 0
        self.epsilon = epsilon
        self.g = lambda x: np.where(
            (x >= self.target_set_b[0]) & (x <= self.target_set_b[1]),
            0,
            -np.log(self.epsilon),
        )

    def get_idx_target_set(self):

        # indices of domain_h in tha target set A
        self.idx_ts_a = np.where(
            (self.domain_h >= self.target_set_a[0]) & (self.domain_h <= self.target_set_a[1])
        )[0]

        # indices of domain_h in tha target set B
        self.idx_ts_b = np.where(
            (self.domain_h >= self.target_set_b[0]) & (self.domain_h <= self.target_set_b[1])
        )[0]

        # indices of the discretized domain corresponding to the target set
        self.idx_ts = np.where(
            ((self.domain_h >= self.target_set_a[0]) & (self.domain_h <= self.target_set_a[1])) |
            ((self.domain_h >= self.target_set_b[0]) & (self.domain_h <= self.target_set_b[1]))
        )[0]

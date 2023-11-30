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

        # problem types flags
        self.is_mgf = False
        self.is_committor = False
        self.overdamped_langevin = False

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
        self.domain_h_flat = x

        # get node indices corresponding to boundary
        is_in_boundary_x = ((x[:, 0] == x_lb) | (x[:, 0] == x_ub))
        is_in_boundary_y = ((x[:, 1] == y_lb) | (x[:, 1] == y_ub))

        self.boundary_x_idx = np.where(is_in_boundary_x == True)[0]
        self.boundary_y_idx = np.where(is_in_boundary_y == True)[0]
        self.boundary_idx = np.where(
            (is_in_boundary_x == True) |
            (is_in_boundary_y == True)
        )[0]
        self.corner_idx = np.where(
            (is_in_boundary_x == True) &
            (is_in_boundary_y == True)
        )[0]

        # get node indices corresponding to the target set
        self.get_target_set_idx()

    def set_mgf_setting(self, lam=1.):
        ''' Set moment generating function of the first hitting time setting
        '''
        # set fht problem flag
        self.is_mgf = True

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

        # running and final costs
        self.f = lambda x: 0
        self.epsilon = epsilon
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
        # flatten domain_h
        x = self.domain_h.reshape(self.Nh, self.d)

        # get index
        self.ts_idx = np.where(self.is_in_target_set_vect(x) == True)[0]

    def get_target_set_idx_committor(self):
        '''
        '''

        # flatten domain_h
        x = self.domain_h.reshape(self.Nh, self.d)

        # indices of domain_h in tha target sets
        self.ts_a_idx = np.where(self.is_in_target_set_a_vect(x) == True)[0]
        self.ts_b_idx = np.where(self.is_in_target_set_b_vect(x) == True)[0]
        self.ts_idx = np.where(
            (self.is_in_target_set_a_vect(x) == True) | (self.is_in_target_set_b_vect(x) == True)
        )[0]

    def get_idx(self, x):
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

    def compute_mfht(self, delta=0.001):
        ''' estimates the expected first hitting time by finite differences of the quantity
            of interest psi(x)
        '''
        from sde_hjb_solver.hjb_solver_2d_st import SolverHJB2D
        from copy import copy

        sde_plus = copy(self)
        sde_plus.set_mgf_setting(lam=delta)
        sol_plus = SolverHJB2D(sde_plus, self.h)
        sol_plus.solve_bvp()

        sde_minus = copy(self)
        sde_minus.set_mgf_setting(lam=-delta)
        sol_minus = SolverHJB2D(sde_minus, self.h)
        sol_minus.solve_bvp()

        return - (sol_plus.psi - sol_minus.psi) / (2 * delta)

    def plot_target_set(self, ylim=None):
        fig, ax = plt.subplots()
        ax.set_title(r'Target set $\mathcal{T}$')
        ax.set_xlabel(r'$x_1$')
        ax.set_ylabel(r'$x_2$')
        ax.set_xlim(self.domain[0])
        ax.set_ylim(self.domain[1])

        # flat domain
        x = self.domain_h.reshape(self.Nh, self.d)

        # target set
        if self.is_mgf:
            set_t = x[self.ts_idx]
            ax.scatter(set_t[:, 0], set_t[:, 1], label=r'$\mathcal{T}$')
        elif self.is_committor:
            set_a = x[self.ts_a_idx]
            set_b = x[self.ts_b_idx]
            ax.scatter(set_a[:, 0], set_a[:, 1], label=r'$A$')
            ax.scatter(set_b[:, 0], set_b[:, 1], label=r'$B$')

        ax.legend(loc='upper right')
        plt.show()

class BrownianMotionCommittor2D(ControlledSDE2D):
    '''
    '''

    def __init__(self, epsilon=1e-10, domain=None, radius_a=1., radius_b=3.):
        super().__init__(domain=domain)

        # log name
        self.name = 'brownian-2d-committor'

        # drift and diffusion terms
        self.drift = lambda x: np.zeros(self.d)
        self.diffusion = 1.

        # domain
        if self.domain is not None:
            self.domain = domain
        else:
            self.domain = np.full((self.d, 2), [-3, 3])

        # target set (in radial coordinates)
        assert radius_a < radius_b, ''
        self.radius_a = radius_a
        self.radius_b = radius_b

        # set in target set condition functions
        self.is_in_target_set_a_vect = lambda x: np.linalg.norm(x, axis=1) < self.radius_a
        self.is_in_target_set_b = lambda x: np.linalg.norm(x) > self.radius_b
        self.is_in_target_set_b_vect = lambda x: np.linalg.norm(x, axis=1) > self.radius_b

        # committor setting
        self.set_committor_setting(epsilon)

    def psi_ana_r_simple(self, r):
        if r < r_a:
            return 0
        elif r < r_b:
            return (np.log(r_a) - np.log(r)) / (np.log(r_a) - np.log(r_b))
        else:
            return 1

    def psi_ana_r(self, r):
        r_a = self.radius_a
        r_b = self.radius_b

        return np.where(
            r < r_a,
            0,
            np.where(r <= r_b, (np.log(r_a) - np.log(r)) / (np.log(r_a) - np.log(r_b)), 1),
        )

    def psi_ana_x(self, x):
        x_norm = np.linalg.norm(x, axis=1)
        return self.psi_ana_r(x_norm)

class OverdampedLangevinSDE2D(ControlledSDE2D):
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

    def plot_2d_potential(self, levels=10, isolines=True, xlim=None, ylim=None):
        fig, ax = plt.subplots()
        ax.set_title(r'Potential $V(x)$')
        ax.set_xlabel(r'$x_1$')
        ax.set_ylabel(r'$x_2$')
        ax.set_xlim(xlim) if xlim is not None else ax.set_xlim(self.domain[0])
        ax.set_xlim(ylim) if ylim is not None else ax.set_ylim(self.domain[1])

        # flat domain
        x = self.domain_h.reshape(self.Nh, self.d)
        pot = self.potential(x).reshape(self.Nx)

        # contour f
        cs = ax.contourf(
            self.domain_h[:, :, 0],
            self.domain_h[:, :, 1],
            pot,
            levels=levels,
            extend='both',
        )
        if isolines: ax.contour(cs, colors='k')

        # colorbar
        cbar = fig.colorbar(cs)
        plt.show()

class DoubleWell2D(OverdampedLangevinSDE2D):
    '''
    '''
    def __init__(self, beta=1., alpha=np.array([1., 1.]), domain=None, ts_pot_level=None):
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

        # target set potential level
        self.ts_pot_level = ts_pot_level

class DoubleWellMGF2D(DoubleWell2D):
    '''
    '''

    def __init__(self, beta=1., alpha=np.array([1., 1.]), lam=1.0, domain=None,
                 ts_pot_level=None, target_set=None):
        super().__init__(beta=beta, alpha=alpha, domain=domain, ts_pot_level=ts_pot_level)

        # log name
        self.name = 'doublewell-2d-mgf__beta{:.1f}_alpha{:.1f}'.format(beta, alpha[0])

        # target set
        if target_set is not None:
            self.target_set = target_set
        else:
            self.target_set = np.full((self.d, 2), [1, 2])

        # set in target set condition function
        self.is_in_target_set = lambda x: (x[:, 0] >= self.target_set[0, 0]) \
                                        & (x[:, 0] <= self.target_set[0, 1]) \
                                        & (x[:, 1] >= self.target_set[1, 0]) \
                                        & (x[:, 1] <= self.target_set[1, 1])
        """
        self.is_in_target_set_vect = lambda x: (self.potential(x) < self.ts_pot_level) \
                                        & (x[:, 0] > 0) & (x[:, 1] > 0)
        """

        # moment generating function of the first hitting time setting
        self.set_mgf_setting(lam=lam)


class DoubleWellCommittor2D(DoubleWell2D):
    '''
    '''

    def __init__(self, beta=1., alpha=np.array([1., 1.]), epsilon=1e-10,
                 domain=None, ts_pot_level=None, target_set_a=None, target_set_b=None):
        super().__init__(beta=beta, alpha=alpha, domain=domain, ts_pot_level=ts_pot_level)

        # log name
        self.name = 'doublewell-2d-committor__beta{:.1f}_alpha{:.1f}'.format(beta, alpha[0])

        # target sets
        if target_set_a is not None:
            self.target_set_a = target_set_a
        else:
            self.target_set_a = np.full((self.d, 2), [-2, -1])

        if target_set_b is not None:
            self.target_set_b = target_set_b
        else:
            self.target_set_b = np.full((self.d, 2), [1, 2])

        # set in target set condition functions
        """
        self.is_in_target_set_a_vect = lambda x: (x[:, 0] >= self.target_set_a[0, 0]) \
                                          & (x[:, 0] <= self.target_set_a[0, 1]) \
                                          & (x[:, 1] >= self.target_set_a[1, 0]) \
                                          & (x[:, 1] <= self.target_set_a[1, 1])

        self.is_in_target_set_b = lambda x: (x[0] >= self.target_set_b[0, 0]) \
                                          & (x[0] <= self.target_set_b[0, 1]) \
                                          & (x[1] >= self.target_set_b[1, 0]) \
                                          & (x[1] <= self.target_set_b[1, 1])


        self.is_in_target_set_b_vect = lambda x: (x[:, 0] >= self.target_set_b[0, 0]) \
                                          & (x[:, 0] <= self.target_set_b[0, 1]) \
                                          & (x[:, 1] >= self.target_set_b[1, 0]) \
                                          & (x[:, 1] <= self.target_set_b[1, 1])
        """
        self.is_in_target_set_a_vect = lambda x: (self.potential(x) < self.ts_pot_level) & \
                                                 (x[:, 0] < 0) & (x[:, 1] < 0)
        self.is_in_target_set_b = lambda x: (self.potential(x) < self.ts_pot_level) & (x[0] > 0) & (x[1] > 0)
        self.is_in_target_set_b_vect = lambda x: (self.potential(x) < self.ts_pot_level) & \
                                                 (x[:, 0] > 0) & (x[:, 1] > 0)

        # committor setting
        self.set_committor_setting(epsilon)


class TripleWell2D(OverdampedLangevinSDE2D):
    ''' Overdamped langevin dynamics following a triple well potential
    '''
    def __init__(self, beta=1., alpha=1., domain=None, ts_pot_level=None):
        super().__init__(beta=beta, domain=domain)

        # potential
        self.alpha = alpha
        self.potential = functools.partial(triple_well_2d, alpha=self.alpha)
        self.gradient = functools.partial(triple_well_gradient_2d, alpha=self.alpha)

        # drift term
        self.drift = lambda x: - self.gradient(x)

        # domain
        if self.domain is not None:
            self.domain = domain
        else:
            self.domain = np.array([[-1.5, 1.5], [-1, 2]])

        # target set potential level
        if ts_pot_level is not None:
            self.ts_pot_level = ts_pot_level
        else:
            self.ts_pot_level = -3.5

class TripleWellMGF2D(TripleWell2D):
    '''
    '''

    def __init__(self, beta=1., alpha=1., lam=1.0, domain=None, ts_pot_level=None):
        super().__init__(beta=beta, alpha=alpha, domain=domain, ts_pot_level=ts_pot_level)

        # log name
        self.name = 'triplewell-2d-mgf__beta{:.1f}_alpha{:.1f}'.format(beta, alpha)

        # set in target set condition function
        self.is_in_target_set_vect = lambda x: (self.potential(x) < self.ts_pot_level) & \
                                               (x[:, 0] > 0)

        # moment generating function of the first hitting time setting
        self.set_mgf_setting(lam=lam)


class TripleWellCommittor2D(TripleWell2D):
    '''
    '''

    def __init__(self, beta=1., epsilon=1e-10, alpha=1., domain=None, ts_pot_level=None):
        super().__init__(beta=beta, alpha=alpha, domain=domain, ts_pot_level=ts_pot_level)

        # log name
        self.name = 'triplewell-2d-committor__beta{:.1f}_alpha{:.1f}'.format(beta, alpha)

        # set in target set condition function
        self.is_in_target_set_a_vect = lambda x: (self.potential(x) < self.ts_pot_level) & \
                                                 (x[:, 0] < 0)
        self.is_in_target_set_b = lambda x: (self.potential(x) < self.ts_pot_level) & (x[0] > 0)
        self.is_in_target_set_b_vect = lambda x: (self.potential(x) < self.ts_pot_level) & \
                                                 (x[:, 0] > 0)

        # committor setting
        self.set_committor_setting(epsilon)

class MuellerBrown2D(OverdampedLangevinSDE2D):
    ''' Overdamped langevin dynamics following the Müller-Brown potential
    '''
    def __init__(self, beta=1., domain=None, ts_pot_level=None):
        super().__init__(beta=beta, domain=domain)

        # potential
        self.potential = mueller_brown_2d
        self.gradient = mueller_brown_gradient_2d

        # drift term
        self.drift = lambda x: - self.gradient(x)

        # domain
        if self.domain is not None:
            self.domain = domain
        else:
            self.domain = np.array([[-1.5, 1], [-0.5, 2]])

          # target set potential level
        if ts_pot_level is not None:
            self.ts_pot_level = ts_pot_level
        else:
            self.ts_pot_level = -100


class MuellerBrownMGF2D(MuellerBrown2D):
    '''
    '''

    def __init__(self, beta=1., lam=1.0, domain=None, ts_pot_level=None):
        super().__init__(beta=beta, domain=domain, ts_pot_level=ts_pot_level)

        # log name
        self.name = 'mueller-brown-2d-mgf__beta{:.1f}'.format(beta)

        # set in target set condition function
        self.is_in_target_set_vect = lambda x: (self.potential(x) < self.ts_pot_level) & \
                                               (x[:, 0] > 0)

        # moment generating function of the first hitting time setting
        self.set_mgf_setting(lam=lam)

class MuellerBrownCommittor2D(MuellerBrown2D):
    '''
    '''

    def __init__(self, beta=1., epsilon=1e-10, domain=None, ts_pot_level=None):
        super().__init__(beta=beta, domain=domain, ts_pot_level=ts_pot_level)

        # log name
        self.name = 'mueller-brown-2d-committor__beta{:.1f}'.format(beta)

        # set in target set condition function
        self.is_in_target_set_a_vect = lambda x: (self.potential(x) < self.ts_pot_level) & \
                                                 (x[:, 0] < 0)
        self.is_in_target_set_b = lambda x: (self.potential(x) < self.ts_pot_level) & (x[0] > 0)
        self.is_in_target_set_b_vect = lambda x: (self.potential(x) < self.ts_pot_level) & \
                                                 (x[:, 0] > 0)

        # committor setting
        self.set_committor_setting(epsilon)

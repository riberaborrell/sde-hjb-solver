import functools

import matplotlib.pyplot as plt
import numpy as np

from sde_hjb_solver.functions import *
from sde_hjb_solver.controlled_sde import ControlledSDE

class ControlledSDE2D(ControlledSDE):
    '''
    '''

    def __init__(self, **kwargs):

        # dimension
        kwargs.update(d=2)

        super().__init__(**kwargs)

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

        self.boundary_x_idx = np.where(is_in_boundary_x)[0]
        self.boundary_y_idx = np.where(is_in_boundary_y)[0]
        self.boundary_idx = np.where(is_in_boundary_x | is_in_boundary_y)[0]
        self.corner_idx = np.where(is_in_boundary_x & is_in_boundary_y)[0]

        # get node indices corresponding to the target set
        self.get_target_set_idx()

    def get_index_vectorized(self, x):
        '''
        '''
        assert x.ndim == 2, ''
        assert x.shape[1] == self.d, ''

        # domain bounds
        lb, ub = self.domain[:, 0], self.domain[:, 1]

        # clip x to domain
        x = np.clip(x, lb, ub)

        # get index by truncation
        idx = np.floor((x - lb) / self.h).astype(int)
        idx = tuple([idx[:, i] for i in range(self.d)])
        return idx


    def get_target_set_idx_mgf(self):
        '''
        '''
        # flatten domain_h
        x = self.domain_h.reshape(self.Nh, self.d)

        # get index
        self.ts_idx = np.where(self.is_target_set_vect(x))[0]

    def get_target_set_idx_committor(self):
        '''
        '''

        # flatten domain_h
        x = self.domain_h.reshape(self.Nh, self.d)

        # indices of domain_h in tha target sets
        self.ts_a_idx = np.where(self.is_target_set_a_vect(x))[0]
        self.ts_b_idx = np.where(self.is_target_set_b_vect(x))[0]
        self.ts_idx = np.where(
            self.is_target_set_a_vect(x) | self.is_target_set_b_vect(x)
        )[0]

    def get_idx(self, x):
        ''' get index of the grid point which approximates x
        '''
        x = np.asarray(x)
        is_scalar = False

        if x.ndim == 0:
            is_scalar = True
            x = x[np.newaxis]

        if x.ndim != 1:
            raise ValueError('x array dimension must be one')

        idx = self.get_idx_truncate(x)

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
        #idx = np.floor((
        #    np.clip(x, self.domain[0], self.domain[1] - 2 * self.h) + self.domain[1]
        #) / self.h).astype(int)


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
        ax.set_title(r'Target set $C$')
        ax.set_xlabel(r'$x_1$')
        ax.set_ylabel(r'$x_2$')
        ax.set_xlim(self.domain[0])
        ax.set_ylim(self.domain[1])

        # flat domain
        x = self.domain_h.reshape(self.Nh, self.d)

        # target set
        if self.setting == 'mgf':
            set_t = x[self.ts_idx]
            ax.scatter(set_t[:, 0], set_t[:, 1], label=r'$C$')
        elif self.setting == 'committor':
            set_a = x[self.ts_a_idx]
            set_b = x[self.ts_b_idx]
            ax.scatter(set_a[:, 0], set_a[:, 1], label=r'$A$')
            ax.scatter(set_b[:, 0], set_b[:, 1], label=r'$B$')

        ax.legend(loc='upper right')
        plt.show()

class BrownianMotion2D(ControlledSDE2D):
    '''
    '''

    def __init__(self, beta: float = 1, **kwargs):
        super().__init__(**kwargs)

        # log name
        self.name = 'brownian-2d-'

        # overdamped langevin flag
        self.is_overdamped_langevin = False

        # drift and diffusion terms
        self.drift = lambda x: np.zeros(self.d)
        self.beta = beta
        self.diffusion = np.sqrt(2 / self.beta)

        # domain
        if self.domain is None:
            self.domain = np.full((self.d, 2), [-1.5, 1.5])

        # parameters string
        self.params_str = 'sigma{:.1f}'.format(self.diffusion)

class BrownianMotionMgf2D(BrownianMotion2D):
    '''
    '''

    def __init__(self, lam=1.0, target_set_r=1., target_set_c=(0., 0.), **kwargs):
        super().__init__(**kwargs)

        # log name
        self.name += 'mgf'

        # target set center and radius
        self.target_set_c = target_set_c
        self.target_set_r = target_set_r

        # first hitting time setting
        self.set_mgf_setting(lam=lam)

        # set in target set condition function
        self.is_target_set = lambda x: np.linalg.norm(x - target_set_c) >= target_set_r
        self.is_target_set_vect = lambda x: np.linalg.norm(x - target_set_c, axis=1) >= target_set_r

    #TODO: generalize to arbitrary scaling factor sigma.
    def psi_ana(self, x):
        c = self.target_set_c
        return np.where(
            self.is_target_set_vect(x),
            1,
            #np.cosh(x) / np.cosh(1),# * 2 /  (np.exp(1) + np.exp(-1)),
            np.cosh(np.sqrt(self.beta) * (x-c)) / np.cosh(np.sqrt(self.beta)),
        )

    #TODO: generalize to arbitrary scaling factor sigma.
    def u_opt_ana(self, x):
        return np.where(
            self.is_target_set(x),
            0,
            self.diffusion * np.tanh(x),
        )

    def mfht_ana(self, x):
        c, r = self.target_set_c, self.target_set_r
        #a, b = c-r, c+r
        return np.where(
            self.is_target_set(x),
            0,
            #- (x - a)*(x - b) / (self.diffusion**2),
            (r**2 - np.linalg.norm(x - self.target_set_c)**2)/(self.diffusion**2),
        )

class BrownianMotionCommittor2D(BrownianMotion2D):
    '''
    '''

    def __init__(self, epsilon=1e-10, radius_a=1., radius_b=3., **kwargs):
        super().__init__(**kwargs)

        # log name
        self.name += 'committor'

        # target set (in radial coordinates)
        assert radius_a < radius_b, ''
        self.radius_a = radius_a
        self.radius_b = radius_b

        # set in target set condition functions
        self.is_target_set_a_vect = lambda x: np.linalg.norm(x, axis=1) < self.radius_a
        self.is_target_set_b = lambda x: np.linalg.norm(x) > self.radius_b
        self.is_target_set_b_vect = lambda x: np.linalg.norm(x, axis=1) > self.radius_b

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

    def __init__(self, beta=1., **kwargs):
        super().__init__(**kwargs)

        # overdamped langevin flag
        self.is_overdamped_langevin = True

        # inverse temperature
        self.beta = beta

        # diffusion
        self.diffusion = np.sqrt(2 / self.beta)

    def plot_2d_potential(self, levels=10, isolines=True, target_set_patch=None, x_init=None,
                          xlim=None, ylim=None):
        fig, ax = plt.subplots()
        ax.set_title(r'Potential $U_{pot}(x)$')
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
            cmap='Blues_r',
        )
        if isolines: ax.contour(cs, colors='k')
        if target_set_patch is not None: ax.add_patch(target_set_patch)
        if x_init is not None:
            ax.plot(x_init[0], x_init[1], marker='x', c='grey', markersize=20, lw=5)

        # colorbar
        cbar = fig.colorbar(cs)
        plt.show()

class DoubleWell2D(OverdampedLangevinSDE2D):
    '''
    '''
    def __init__(self, alpha=np.array([1., 1.]), ts_pot_level=None, **kwargs):
        super().__init__(**kwargs)

        # check alpha
        assert alpha.size == 2, 'alpha must be an array of size 2'

        # log name
        self.name = 'doublewell-2d-'
        self.params_str = 'beta{:.1f}_alpha-i{:.1f}_alpha-j{:.1f}' \
                          ''.format(self.beta, alpha[0], alpha[1])

        # potential
        self.alpha = alpha
        self.potential = functools.partial(double_well, alpha=self.alpha)
        self.gradient = functools.partial(double_well_gradient, alpha=self.alpha)

        # drift term
        self.drift = lambda x: - self.gradient(x)

        # domain
        if self.domain is None:
            self.domain = np.full((self.d, 2), [-2, 2])

        # target set potential level
        self.ts_pot_level = ts_pot_level

class DoubleWellMgf2D(DoubleWell2D):
    '''
    '''

    def __init__(self, lam=1.0, target_set=None, **kwargs):
        super().__init__(**kwargs)

        # log name
        self.name += 'mgf'

        """
        # rectangle target set
        if target_set is not None:
            self.target_set = target_set
        else:
            self.target_set = np.full((self.d, 2), [1, 2])

        # set in target set condition function
        self.is_target_set_vect = lambda x: (x[:, 0] >= self.target_set[0, 0]) \
                                            & (x[:, 0] <= self.target_set[0, 1]) \
                                            & (x[:, 1] >= self.target_set[1, 0]) \
                                            & (x[:, 1] <= self.target_set[1, 1])
        """

        # set in target set condition function
        self.is_target_set_vect = lambda x: (self.potential(x) < self.ts_pot_level) \
                                          & (x[:, 0] > 0) & (x[:, 1] > 0)

        # moment generating function of the first hitting time setting
        self.set_mgf_setting(lam=lam)


class DoubleWellCommittor2D(DoubleWell2D):
    '''
    '''

    def __init__(self, epsilon=1e-10, target_set_a=None, target_set_b=None, **kwargs):
        super().__init__(**kwargs)

        # log name
        self.name += 'committor'

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
        self.is_target_set_a_vect = lambda x: (x[:, 0] >= self.target_set_a[0, 0]) \
                                          & (x[:, 0] <= self.target_set_a[0, 1]) \
                                          & (x[:, 1] >= self.target_set_a[1, 0]) \
                                          & (x[:, 1] <= self.target_set_a[1, 1])

        self.is_target_set_b = lambda x: (x[0] >= self.target_set_b[0, 0]) \
                                          & (x[0] <= self.target_set_b[0, 1]) \
                                          & (x[1] >= self.target_set_b[1, 0]) \
                                          & (x[1] <= self.target_set_b[1, 1])


        self.is_target_set_b_vect = lambda x: (x[:, 0] >= self.target_set_b[0, 0]) \
                                          & (x[:, 0] <= self.target_set_b[0, 1]) \
                                          & (x[:, 1] >= self.target_set_b[1, 0]) \
                                          & (x[:, 1] <= self.target_set_b[1, 1])
        """
        self.is_target_set_a_vect = lambda x: (self.potential(x) < self.ts_pot_level) & \
                                                 (x[:, 0] < 0) & (x[:, 1] < 0)
        self.is_target_set_b = lambda x: (self.potential(x) < self.ts_pot_level) & (x[0] > 0) & (x[1] > 0)
        self.is_target_set_b_vect = lambda x: (self.potential(x) < self.ts_pot_level) & \
                                                 (x[:, 0] > 0) & (x[:, 1] > 0)

        # committor setting
        self.set_committor_setting(epsilon)


class TripleWell2D(OverdampedLangevinSDE2D):
    ''' Overdamped langevin dynamics following a triple well potential
    '''
    def __init__(self, alpha=np.array([1.]), ts_pot_level=-3.5, **kwargs):
        super().__init__(**kwargs)

        # check alpha
        assert alpha.size == 1, 'alpha must be an array of size 1'
        self.alpha = alpha[0]

        # log name
        self.name = 'triplewell-2d-'
        self.params_str = 'beta{:.1f}_alpha{:.1f}'.format(self.beta, self.alpha)

        # potential
        self.potential = functools.partial(triple_well_2d, alpha=self.alpha)
        self.gradient = functools.partial(triple_well_gradient_2d, alpha=self.alpha)

        # drift term
        self.drift = lambda x: - self.gradient(x)

        # domain
        if self.domain is None:
            self.domain = np.array([[-1.5, 1.5], [-1, 2]])

        # target set potential level
        self.ts_pot_level = ts_pot_level

class TripleWellMgf2D(TripleWell2D):
    '''
    '''

    def __init__(self, lam=1.0, **kwargs):
        super().__init__(**kwargs)

        # log name
        self.name += 'mgf'

        # set in target set condition function
        self.is_target_set_vect = lambda x: (self.potential(x) < self.ts_pot_level) & \
                                               (x[:, 0] > 0)

        # moment generating function of the first hitting time setting
        self.set_mgf_setting(lam=lam)


class TripleWellCommittor2D(TripleWell2D):
    '''
    '''

    def __init__(self, epsilon=1e-10, **kwargs):
        super().__init__(**kwargs)

        # log name
        self.name += 'committor'

        # set in target set condition function
        self.is_target_set_a_vect = lambda x: (self.potential(x) < self.ts_pot_level) & \
                                                 (x[:, 0] < 0)
        self.is_target_set_b = lambda x: (self.potential(x) < self.ts_pot_level) & (x[0] > 0)
        self.is_target_set_b_vect = lambda x: (self.potential(x) < self.ts_pot_level) & \
                                                 (x[:, 0] > 0)

        # committor setting
        self.set_committor_setting(epsilon)

class MuellerBrown2D(OverdampedLangevinSDE2D):
    ''' Overdamped langevin dynamics following the Müller-Brown potential
    '''
    def __init__(self, ts_pot_level=-100, **kwargs):
        super().__init__(**kwargs)

        # log name
        self.name = 'mueller-brown-2d-'
        self.params_str = 'beta{:.1f}'.format(self.beta)

        # potential
        self.potential = mueller_brown_2d
        self.gradient = mueller_brown_gradient_2d

        # drift term
        self.drift = lambda x: - self.gradient(x)

        # domain
        if self.domain is None:
            self.domain = np.array([[-1.5, 1], [-0.5, 2]])

        # alternative domain
        # domain = x \in R^2 | V(x) <= 250

        # target set potential level
        self.ts_pot_level = ts_pot_level

        # a = (−0.558, 1.441)
        # b = (0.623, 0.028)


class MuellerBrownMgf2D(MuellerBrown2D):
    '''
    '''

    def __init__(self, lam=1.0, **kwargs):
        super().__init__(**kwargs)

        # log name
        self.name += 'mgf'

        # set in target set condition function
        self.is_target_set_vect = lambda x: (self.potential(x) < self.ts_pot_level) & \
                                               (x[:, 0] > 0)

        # moment generating function of the first hitting time setting
        self.set_mgf_setting(lam=lam)

class MuellerBrownCommittor2D(MuellerBrown2D):
    '''
    '''

    def __init__(self, epsilon=1e-10, **kwargs):
        super().__init__(**kwargs)

        # log name
        self.name += 'committor'

        # set in target set condition function
        self.is_target_set_a_vect = lambda x: (self.potential(x) < self.ts_pot_level) & \
                                                 (x[:, 0] < 0)
        self.is_target_set_b = lambda x: (self.potential(x) < self.ts_pot_level) & (x[0] > 0)
        self.is_target_set_b_vect = lambda x: (self.potential(x) < self.ts_pot_level) & \
                                                 (x[:, 0] > 0)

        # committor setting
        self.set_committor_setting(epsilon)

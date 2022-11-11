import time

import numpy as np
import matplotlib.pyplot as plt

from controlled_sde_1d import ControlledSDE1D

class SolverHJB1D(object):
    ''' This class provides a solver of the following 1d BVP by using a
        finite differences method:
            0 = LΨ − f Ψ in S
            Ψ = exp(− g) in ∂S,
        where f = 1, g = 1 and L is the infinitessimal generator
        of the not controlled 1d diffusion process:
            L = b(x) d/dx + 1/2 sigma^2 d^2/dx^2

    Attributes
    ----------
    sde: ControlledSDE object
        controlled sde object
    ct_initial: float
        initial computational time
    ct_time: float
        final computational time
    ct: float
        computational time
    psi: array
        solution of the BVP problem
    solved: bool
        flag telling us if the problem is solved
    value_function: array
        value function of the HJB equation
    u_opt: array
        optimal control of the HJB equation
    mfht: array
       mean first hitting time


    Methods
    -------
    __init__(sde, h)

    start_timer()

    stop_timer()

    get_x(k)

    solve_bvp()

    compute_value_function()

    compute_optimal_control()

    compute_mfht()

    save()

    load()

    get_psi_at_x(x)

    get_value_function_at_x(x)

    get_u_opt_at_x(x)

    write_report(x)

    plot_1d_psi(ylim=None)

    plot_1d_value_function(ylim=None)

    plot_1d_controlled_potential(ylim=None)

    plot_1d_control(ylim=None)

    plot_1d_controlled_drift(ylim=None)

    plot_1d_exp_fht()

    '''

    def __init__(self, sde, h):
        ''' init method

        Parameters
        ----------
        sde: langevinSDE object
            overdamped langevin sde object
        h: float
            step size

        Raises
        ------
        NotImplementedError
            If dimension d is greater than 1
        '''

        if sde.d != 1:
            raise NotImplementedError('d > 1 not supported')

        # sde object
        self.sde = sde

        # discretization step
        self.h = h

        # dir_path
        #self.dir_path = get_hjb_solution_dir_path(sde.settings_dir_path, h)

    def start_timer(self):
        ''' start timer
        '''
        self.ct_initial = time.perf_counter()

    def stop_timer(self):
        ''' stop timer
        '''
        self.ct_final = time.perf_counter()
        self.ct = self.ct_final - self.ct_initial


    def get_x(self, k):
        ''' returns the x-coordinate of the node k

        Parameters
        ----------
        k: int
            index of the node

        Returns
        -------
        float
            point in the domain
        '''
        assert k in np.arange(self.sde.Nh), ''

        return self.sde.domain_h[k]

    def solve_bvp(self):
        ''' solve bvp by using finite difference
        '''

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
        idx_boundary = np.array([0, self.sde.Nh - 1])

        # nodes in target set
        idx_ts = self.sde.idx_ts

        for k in np.arange(self.sde.Nh):

            # assemble matrix A and vector b on S
            if k not in idx_ts and k not in idx_boundary:
                x = self.get_x(k)

                # drift and diffusion
                #drift = self.sde.drift(x, self.sde.alpha)
                drift = self.sde.drift(x)
                sigma = self.sde.diffusion

                A[k, k] = - sigma**2 / h**2 - self.sde.f(x)
                A[k, k - 1] = sigma**2 / (2 * h**2) + drift / (2 * h)
                A[k, k + 1] = sigma**2 / (2 * h**2) - drift / (2 * h)
                b[k] = 0

            # impose condition on ∂S
            elif k in idx_ts:
                x = self.get_x(k)
                A[k, k] = 1
                b[k] = np.exp(- self.sde.g(x))

        # stability condition on the boundary: Psi should be flat

        # Psi_0 = Psi_1
        A[0, 0] = 1
        A[0, 1] = -1
        b[0] = 0

        # psi_{Nh-1} = Psi_N)
        A[-1, -1] = 1
        A[-1, -2] = -1
        b[-1] = 0

        # solve linear system and save
        self.psi = np.linalg.solve(A, b)
        self.solved = True

        # stop timer
        self.stop_timer()

    def compute_value_function(self):
        ''' this methos computes the value function
                value_f = - log (psi)
        '''
        self.value_function =  - np.log(self.psi)

    def compute_optimal_control(self):
        ''' this method computes by finite differences the optimal control
                u_opt = - sigma ∇_x value_f
        '''
        assert hasattr(self, 'value_function'), ''
        assert self.value_function.ndim == self.sde.d, ''

        # diffusion term
        sigma = self.sde.diffusion

        self.u_opt = np.zeros(self.sde.Nh)

        # central difference approximation
        # for any k in {1, ..., Nh-2}
        # u_opt(x_k) = - sigma (Phi_{k+1} - Phi_{k-1}) / 2h 

        self.u_opt[1: -1] = - sigma \
                          * (self.value_function[2:] - self.value_function[:-2]) \
                          / (2 * self.sde.h)
        self.u_opt[0] = self.u_opt[1]
        self.u_opt[-1] = self.u_opt[-2]

    def compute_mfht(self):
        ''' computes the expected first hitting time by finite differences of the quantity
            of interest psi(x)
        '''
        from copy import copy

        l = 0.01

        sde_plus = copy(self.sde)
        sde_plus.sigma = l
        sde_plus.f = lambda x: sde_plus.sigma * sde_plus.beta
        sol_plus = SolverHJB1D(sde_plus, self.h)
        sol_plus.solve_bvp()

        sde_minus = copy(self.sde)
        sde_minus.sigma = -l
        sde_minus.f = lambda x: sde_minus.sigma * sde_minus.beta
        sol_minus = SolverHJB1D(sde_minus, self.h)
        sol_minus.solve_bvp()

        partial_psi = (sol_plus.psi - sol_minus.psi) / (2 * l)
        self.mfht = - partial_psi / self.sde.beta


    def save(self):
        ''' saves some attributes as arrays into a .npz file
        '''
        # create directoreis of the given path if it does not exist
        if not os.path.isdir(self.dir_path):
            os.makedirs(self.dir_path)

        # save arrays in a npz file
        np.savez(
            os.path.join(self.dir_path, 'hjb-solution-1d.npz'),
            domain_h=self.sde.domain_h,
            Nx=self.sde.Nx,
            Nh=self.sde.Nh,
            psi=self.psi,
            value_function=self.value_function,
            u_opt=self.u_opt,
            ct=self.ct,
        )

    def load(self):
        ''' loads the saved arrays and sets them as attributes back
        '''
        try:
            data = np.load(
                os.path.join(self.dir_path, 'hjb-solution-1d.npz'),
                allow_pickle=True,
            )
            for attr_name in data.files:

                # get attribute from data
                if data[attr_name].ndim == 0:
                    attr = data[attr_name][()]
                else:
                    attr = data[attr_name]

                # langevin SDE attribute
                if attr_name in ['domain_h', 'Nx', 'Nh']:

                    # if attribute exists check if they are the same
                    if hasattr(self.sde, attr_name):
                        assert getattr(self.sde, attr_name) == attr

                    # if attribute does not exist save attribute
                    else:
                        setattr(self.sde, attr_name, attr)

                # hjb solver attribute
                else:

                    # if attribute exists check if they are the same
                    if hasattr(self, attr_name):
                        assert getattr(self, attr_name) == attr

                    # if attribute does not exist save attribute
                    else:
                        setattr(self, attr_name, attr)

            return True

        except:
            print('no hjb-solution found with h={:.0e}'.format(self.h))
            return False

    def get_psi_at_x(self, x):
        ''' evaluates solution of the BVP at x

        Parameters
        ----------
        x: array
            point in the domain

        Returns
        -------
        float
            psi at x
        '''
        # get index of x
        idx = self.sde.get_index(x)

        # evaluate psi at idx
        return self.psi[idx] if hasattr(self, 'psi') else None

    def get_value_function_at_x(self, x):
        ''' evaluates the value function at x

        Parameters
        ----------
        x: array
            point in the domain

        Returns
        -------
        float
            value function at x
        '''
        # get index of x
        idx = self.sde.get_index(x)

        # evaluate psi at idx
        return self.value_f[idx] if hasattr(self, 'value_f') else None

    def get_u_opt_at_x(self, x):
        ''' evaluates the optimal control at x

        Parameters
        ----------
        x: array
            point in the domain

        Returns
        -------
        array
            optimal control at x
        '''
        # get index of x
        idx = self.sde.get_index(x)

        # evaluate psi at idx
        return self.u_opt[idx] if hasattr(self, 'u_opt') else None

    def get_controlled_potential_and_drift(self):
        ''' computes the potential, bias potential, controlled potential, gradient,
            controlled drift
        '''

        # flatten domain_h
        x = np.expand_dims(self.sde.domain_h, axis=1)

        # potential, bias potential and tilted potential
        self.V = np.squeeze(self.sde.potential(x))
        self.bias_potential = 2 * self.value_function / self.sde.beta
        self.controlled_potential = self.V + self.bias_potential

        # gradient and tilted drift
        self.dV = np.squeeze(self.sde.gradient(x))
        #self.dV = self.sde.drift(x, self.sde.alpha)
        self.controlled_drift = - self.dV + np.sqrt(2) * self.u_opt


    def write_report(self, x):
        ''' writes the hjb solver parameters

        Parameters
        ----------
        x: array
            point in the domain

        '''
        from utils import get_time_in_hms

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
        u_opt_max = self.u_opt[idx_u_max]
        print('argmax_x u_opt: {:2.3f}'.format(x_u_max))
        print('max u_opt(x): {:2.3f}'.format(x_u_max))

        # computational time
        h, m, s = get_time_in_hms(self.ct)
        print('\nComputational time: {:d}:{:02d}:{:02.2f}\n'.format(h, m, s))

    def plot_1d_psi(self, ylim=None):
        fig, ax = plt.subplots()
        ax.set_xlabel('x')
        ax.set_xlim(self.sde.domain)
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.plot(self.sde.domain_h, self.psi)
        plt.show()

    def plot_1d_value_function(self, ylim=None):
        fig, ax = plt.subplots()
        ax.set_xlabel('x')
        ax.set_xlim(self.sde.domain)
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.plot(self.sde.domain_h, self.value_function)
        plt.show()

    def plot_1d_controlled_potential(self, ylim=None):
        fig, ax = plt.subplots()
        ax.set_xlabel('x')
        ax.set_xlim(self.sde.domain)
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.plot(self.sde.domain_h, self.V)
        ax.plot(self.sde.domain_h, self.controlled_potential)
        plt.show()

    def plot_1d_control(self, ylim=None):
        fig, ax = plt.subplots()
        ax.set_xlabel('x')
        ax.set_xlim(self.sde.domain)
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.plot(self.sde.domain_h, self.u_opt)
        plt.show()

    def plot_1d_controlled_drift(self, ylim=None):
        fig, ax = plt.subplots()
        ax.set_xlabel('x')
        ax.set_xlim(self.sde.domain)
        if ylim is not None:
            ax.set_ylim(ylim)
        self.get_controlled_potential_and_drift()
        ax.plot(self.sde.domain_h, self.controlled_drift)
        plt.show()

    def plot_1d_mfht(self, ylim=None):
        fig, ax = plt.subplots()
        ax.set_xlabel('x')
        ax.set_xlim(self.sde.domain)
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.plot(self.sde.domain_h, self.mfht)
        plt.show()

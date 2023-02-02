import os
import time

import matplotlib.pyplot as plt
import numpy as np

from sde_hjb_solver.utils_path import make_dir_path, save_data, load_data

class SolverHJB1D(object):
    ''' This class provides a solver of the following 1d BVP by using a
        finite differences method:
            0 = LΨ − f Ψ in S
            Ψ = exp(− g) in ∂S,
        where f and g are the running and terminal costs of the work functional,
        the solution Ψ is the quantity we want to estimate and L is the infinitessimal
        generator of the not controlled 1d diffusion process:
            L = b(x) d/dx + 1/2 sigma(x)^2 d^2/dx^2

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

    save()

    load()

    get_psi_at_x(x)

    get_value_function_at_x(x)

    get_u_opt_at_x(x)

    get_perturbed_potential_and_drift()

    write_report(x)

    plot_1d_psi(ylim=None)

    plot_1d_value_function(ylim=None)

    plot_1d_perturbed_potential(ylim=None)

    plot_1d_control(ylim=None)

    plot_1d_perturbed_drift(ylim=None)

    plot_1d_mfht(ylim=None)

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

        # rel directory path
        self.rel_dir_path = os.path.join(sde.name, 'h{:.0e}'.format(h))

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

            # get point
            x = self.get_x(k)

            # assemble matrix A and vector b on S
            if k not in idx_ts and k not in idx_boundary:

                # drift and diffusion at x
                drift = self.sde.drift(x)
                sigma = self.sde.diffusion

                A[k, k] = - sigma**2 / h**2 - self.sde.f(x)
                A[k, k - 1] = sigma**2 / (2 * h**2) - drift / (2 * h)
                A[k, k + 1] = sigma**2 / (2 * h**2) + drift / (2 * h)

            # impose condition on ∂S
            elif k in idx_ts:
                A[k, k] = 1
                b[k] = np.exp(- self.sde.g(x))

            # stability condition on the boundary: Psi should be flat
            elif k in idx_boundary:
                if k == 0:
                    # Psi_0 = Psi_1
                    A[0, 0] = 1
                    A[0, 1] = -1

                if k == self.sde.Nh - 1:
                    # psi_{Nh-1} = Psi_N)
                    A[-1, -1] = 1
                    A[-1, -2] = -1

        # solve linear system and save
        self.psi = np.linalg.solve(A, b)
        self.solved = True

        # stop timer
        self.stop_timer()

    def compute_value_function(self):
        ''' this method computes the value function
                phi = - log (psi)
        '''
        self.value_function =  - np.log(self.psi)

    def compute_optimal_control(self):
        ''' this method computes by finite differences the optimal control
                u_opt = - sigma d/dx phi
        '''
        assert hasattr(self, 'value_function'), ''
        assert self.value_function.ndim == self.sde.d, ''

        # central difference approximation
        # for any k in {1, ..., Nh-2}
        # u_opt(x_k) = - sigma (Phi_{k+1} - Phi_{k-1}) / 2h 

        # diffusion term
        sigma = self.sde.diffusion

        # preallocate u_opt
        self.u_opt = np.zeros(self.sde.Nh)

        self.u_opt[1: -1] = - sigma \
                          * (self.value_function[2:] - self.value_function[:-2]) \
                          / (2 * self.sde.h)
        self.u_opt[0] = self.u_opt[1]
        self.u_opt[-1] = self.u_opt[-2]

    def save(self):
        ''' saves some attributes as arrays into a .npz file
        '''
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

        # save arrays in a npz file
        save_data(data, self.rel_dir_path)

    def load(self):
        ''' loads the saved arrays and sets them as attributes back
        '''
        data = load_data(self.rel_dir_path)
        try:
            for attr_name in data.keys():

                # get attribute from data
                attr = data[attr_name]

                # Controlled SDE attribute
                if attr_name in ['h', 'domain_h', 'Nx', 'Nh']:

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
            print('Attribute to load already exists and does not match')
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

        # evaluate psi at x
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

        # evaluate value function at x
        return self.value_function[idx] if hasattr(self, 'value_function') else None

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

        # evaluate optimal control at x
        return self.u_opt[idx] if hasattr(self, 'u_opt') else None

    def get_perturbed_potential_and_drift(self):
        ''' computes the potential, bias potential, controlled potential, gradient,
            controlled drift
        '''

        # flatten domain_h
        x = np.expand_dims(self.sde.domain_h, axis=1)

        # diffusion term
        sigma = self.sde.diffusion

        # potential, bias potential and tilted potential
        self.V = np.squeeze(self.sde.potential(x))
        self.bias_potential = (sigma**2) * self.value_function
        self.perturbed_potential = self.V + self.bias_potential

        # gradient and tilted drift
        self.dV = np.squeeze(self.sde.gradient(x))
        self.perturbed_drift = - self.dV + sigma * self.u_opt


    def write_report(self, x):
        ''' writes the hjb solver parameters and the value of the solution at x

        Parameters
        ----------
        x: array
            point in the domain
        '''
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
        u_opt_max = self.u_opt[idx_u_max]
        print('argmax_x u_opt(x): {:2.3f}'.format(x_u_max))
        print('max_x u_opt(x): {:2.3f}'.format(u_opt_max))

        # computational time
        h, m, s = get_time_in_hms(self.ct)
        print('\nComputational time: {:d}:{:02d}:{:02.2f}\n'.format(h, m, s))

    def plot_1d_psi(self, ylim=None):
        fig, ax = plt.subplots()
        ax.set_title(r'Estimation of $\Psi(x)$')
        ax.set_xlabel('x')
        ax.set_xlim(self.sde.domain)
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.plot(self.sde.domain_h, self.psi)
        plt.show()

    def plot_1d_value_function(self, ylim=None):
        fig, ax = plt.subplots()
        ax.set_title(r'Estimation of $\Phi(x)$')
        ax.set_xlabel('x')
        ax.set_xlim(self.sde.domain)
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.plot(self.sde.domain_h, self.value_function)
        plt.show()

    def plot_1d_perturbed_potential(self, ylim=None):
        fig, ax = plt.subplots()
        ax.set_title(r'Perturbed potential $(V + V_{bias})(x)$')
        ax.set_xlabel('x')
        ax.set_xlim(self.sde.domain)
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.plot(self.sde.domain_h, self.V)
        ax.plot(self.sde.domain_h, self.perturbed_potential)
        plt.show()

    def plot_1d_control(self, ylim=None):
        fig, ax = plt.subplots()
        ax.set_title(r'Optimal control $u^*(x)$')
        ax.set_xlabel('x')
        ax.set_xlim(self.sde.domain)
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.plot(self.sde.domain_h, self.u_opt)
        plt.show()

    def plot_1d_perturbed_drift(self, ylim=None):
        fig, ax = plt.subplots()
        ax.set_title(r'Perturbed drift $\nabla(V + V_{bias})(x)$')
        ax.set_xlabel('x')
        ax.set_xlim(self.sde.domain)
        if ylim is not None:
            ax.set_ylim(ylim)
        self.get_perturbed_potential_and_drift()
        ax.plot(self.sde.domain_h, self.perturbed_drift)
        plt.show()

    def plot_1d_mfht(self, ylim=None):
        fig, ax = plt.subplots()
        ax.set_title(r'Estimation of $\mathbb{E}^x[\tau]$')
        ax.set_xlabel('x')
        ax.set_xlim(self.sde.domain)
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.plot(self.sde.domain_h, self.mfht)
        plt.show()

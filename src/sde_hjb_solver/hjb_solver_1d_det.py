import time

import matplotlib.pyplot as plt
import numpy as np

import scipy.sparse as sparse
import scipy.sparse.linalg as linalg
from scipy.linalg import solve_banded

from sde_hjb_solver.utils import arange_generator


class SolverHJB1DDet(object):
    ''' This class provides a solver of the following 1d BVP by using a
        finite differences method:
            ∂_t Ψ  = LΨ − f Ψ  for all x \in \R^d, t \in [0, T)
            Ψ(T, x) = exp(− g(x)) for all x \in \R^d
        where f and g are the running and terminal costs of the work functional,
        the solution Ψ is the quantity we want to estimate and L is the infinitessimal
        generator of the not controlled 1d diffusion process:
            L = b(x) d/dx + 1/2 sigma(x)^2 d^2/dx^2
   '''

    def __init__(self, sde, h, dt, T):


        if sde.d != 1:
            raise NotImplementedError('d > 1 not supported')

        # sde object
        self.sde = sde

        # space discretization step
        self.h = h

        # time discretization step
        self.dt = dt

        # final time
        self.T = T

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

    def solve_bvp_det(self):
        ''' solve bvp by using finite difference
        '''

        # start timer
        self.start_timer()

        # space discretization step
        h = self.h

        # time discretization step
        dt = self.dt

        # discretize domain
        self.sde.discretize_domain_1d(h)

        # preallocate psi
        self.K = int(self.T / self.dt)
        self.psi = np.empty((self.K + 1, + self.sde.Nh))

        # evaluate psi at T
        x = np.expand_dims(self.sde.domain_h, axis=1)
        self.psi[self.K, :] = np.exp(- self.sde.g(x))

        # reverse loop over the time step indices
        for l in range(self.K - 1, -1, -1):

            # assemble linear system of equations: \psi^{l-1} = A \psi^l
            #A = sparse.lil_matrix((self.sde.Nh, self.sde.Nh))
            A = np.zeros((self.sde.Nh, self.sde.Nh))

            # nodes in boundary
            idx_boundary = np.array([0, self.sde.Nh - 1])

            #for k in arange_generator(self.Nh):
            for k in np.arange(self.sde.Nh):

                # get point
                x = self.get_x(k)

                # assemble matrix A and vector b on the domain
                if k not in idx_boundary:

                    # drift and diffusion
                    drift = self.sde.drift(x)
                    sigma = self.sde.diffusion

                    # forward time scheme
                    #TODO: check drift signs
                    A[k, k] = 1 + dt * sigma**2 / (h**2) + dt * self.sde.f(x)
                    A[k, k - 1] = - dt * sigma**2 / (2 * h**2) - dt * drift / (2 * h)
                    A[k, k + 1] = - dt * sigma**2 / (2 * h**2) + dt * drift / (2 * h)

                    # backward time scheme
                    #A[k, k] = 1 - dt * sigma**2 / (h**2) - dt * self.sde.f(x)
                    #A[k, k - 1] = + dt * sigma**2 / (2 * h**2) + dt * drift / (2 * h)
                    #A[k, k + 1] = + dt * sigma**2 / (2 * h**2) - dt * drift / (2 * h)

                ## stability condition on the boundary: Psi should be flat
                #else:
                #    if k == 0:
                #        # Psi_0^l = Psi_1^l ?
                #        #A[0, 0] = 1
                #        #A[0, 1] = - 1

                #        # Psi_0^{l-1} = Psi_0^l + \Delta t Psi_1^l ?
                #        #A[0, 0] = 1
                #        #A[0, 1] = dt


                #    elif k == self.sde.Nh -1:
                #        # psi_{Nh-1} = Psi_N) ?
                #        #A[-1, -1] = 1
                #        #A[-1, -2] = -1

                #        # Psi_N^{l-1} = Psi_N^l + \Delta t Psi_{N-1}^l ?
                #        A[-1, -1] = 1
                #        A[-1, -2] = dt

            # stability condition on the boundary: Psi should be flat
            # forward time scheme
            #A[0, 0] = A[1, 0]
            #A[0, 1] = A[1, 1]
            #A[0, 2] = A[1, 2]
            #A[-1, -1] = A[-2, -1]
            #A[-1, -2] = A[-2, -2]
            #A[-1, -3] = A[-2, -3]

            # backward time scheme
            #A[0, 0] = A[1, 0]
            #A[0, 1] = A[1, 1]
            #A[0, 2] = A[1, 2]
            #A[-1, -1] = A[-2, -1]
            #A[-1, -2] = A[-2, -2]
            #A[-1, -3] = A[-2, -3]

            # forwad time scheme
            #A_inv = np.linalg.inv(A)
            #self.psi[l, :] = np.dot(A_inv, self.psi[l+1, :])
            A_left_inv = np.linalg.pinv(A[:, 1:-1])
            self.psi[l, 1:-1] = np.dot(A_left_inv, self.psi[l+1, :])
            self.psi[l, 0] = self.psi[l, 1]
            self.psi[l, -1] = self.psi[l, -2]

            # backward time scheme
            #self.psi[l, :] = np.dot(A, self.psi[l+1, :])
            #self.psi[l, 1:-1] = np.dot(A[1:-1, :], self.psi[l+1, :])
            #self.psi[l, 0] = self.psi[l, 1]
            #self.psi[l, -1] = self.psi[l, -2]

        self.solved = True

        # stop timer
        self.stop_timer()

    def solve_bvp_det_eigenproblem(self):
        ''' solve bvp eigenproblem
        '''
        # start timer
        self.start_timer()

        # space discretization step
        h = self.h

        # time discretization step
        dt = self.dt

        # discretize domain
        self.sde.discretize_domain_1d(h)

        # preallocate psi
        self.K = int(self.T / self.dt)
        self.psi = np.empty((self.K + 1, + self.sde.Nh))

        # lower bound
        lb = self.sde.domain[0]

        # A
        A = np.zeros([self.sde.Nh, self.sde.Nh])
        for k in arange_generator(self.sde.Nh):

            x = lb + (k + 0.5) * h
            if k > 0:
                x0 = lb + (k - 0.5) * h
                x1 = lb + k * h
                A[k, k - 1] = - np.exp(
                    self.sde.beta * 0.5 * (
                        self.sde.potential(x0) + self.sde.potential(x) - 2 * self.sde.potential(x1)
                    )
                ) / h ** 2
                A[k, k] = np.exp(
                    self.sde.beta * (self.sde.potential(x) - self.sde.potential(x1))
                ) / h**2

            if k < self.sde.Nh - 1:
                x0 = lb + (k + 1.5) * h
                x1 = lb + (k + 1) * h
                A[k, k + 1] = - np.exp(
                    self.sde.beta * 0.5 * (
                        self.sde.potential(x0) + self.sde.potential(x) - 2 * self.sde.potential(x1)
                    )
                ) / h ** 2
                A[k, k] = A[k, k] + np.exp(
                    self.sde.beta * (self.sde.potential(x) - self.sde.potential(x1))
                ) / h ** 2

        A = - A / self.sde.beta

        x = np.expand_dims(self.sde.domain_h, axis=1)
        D = np.diag(np.exp(self.sde.beta * self.sde.potential(x) / 2))
        D_inv = np.diag(np.exp(-self.sde.beta * self.sde.potential(x) / 2))

        np.linalg.cond(np.eye(self.sde.Nh) - dt * A)
        #w, vv = np.linalg.eigh(np.eye(self.nx) - self.dt * A)

        # evaluate psi at T
        self.psi[self.K, :] = np.exp(- self.sde.g(x))

        for l in range(self.K - 1, -1, -1):
            band = - dt * np.vstack([
                np.append([0], np.diagonal(A, offset=1)),
                np.diagonal(A, offset=0) - self.K / self.T,
                np.append(np.diagonal(A, offset=1), [0])
            ])

            self.psi[l, :] = D.dot(solve_banded([1, 1], band, D_inv.dot(self.psi[l + 1, :])))
            #psi[n, :] = np.dot(D, np.linalg.solve(np.eye(self.nx) - dt * A, D_inv.dot(psi[n + 1, :])));

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
                u_opt(t, x) = - sigma ∂/∂x phi(t, x)
        '''
        assert hasattr(self, 'value_function'), ''
        assert self.value_function.ndim == 2, ''

        # central difference approximation
        # for any l in {0, K} and any k in {1, ..., Nh-2}
        # u_opt(x_k, t_l) = - sigma (Phi_{k+1}^l - Phi_{k-1}^l) / 2h 

        # preallocate u_opt at each time step
        self.u_opt = np.zeros((self.K + 1, self.sde.Nh))

        # diffusion term
        sigma = self.sde.diffusion

        # for each time step
        for l in range(self.K + 1):

            self.u_opt[l, 1: -1] = - sigma \
                              * (self.value_function[l, 2:] - self.value_function[l, :-2]) \
                              / (2 * self.sde.h)
            self.u_opt[l, 0] = self.u_opt[l, 1]
            self.u_opt[l, -1] = self.u_opt[l, -2]

        #for k in range(self.sde.Nh - 1):

                #self.u_opt_i[l, k] = - (np.sqrt(2) / self.beta) * (
                #    - np.log(self.psi_i[l, k + 1])
                #    + np.log(self.psi_i[l, k])
                #) / self.h
        #self.u = 2 / beta * np.gradient(np.log(self.psi), self.dx, 1)


    def save(self):
        ''' saves some attributes as arrays into a .npz file
        '''
        # create data dictionary 
        data = {
            'h': self.h,
            #'domain_h': self.sde.domain_h,
            'Nx': self.sde.Nx,
            'Nh': self.sde.Nh,
            'x': self.x,
            'T': self.T,
            'dt': self.dt,
            'K': self.K,
            'psi_i': self.psi_i,
            'u_opt_i': self.u_opt_i,
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
            print('Attribute to load already exists and does not match')
            return False

    def get_time_index(self, t):
        ''' returns time index for the given time t
        '''
        assert 0 <= t <= self.T, ''

        return int(np.ceil(t / self.dt))

    def get_psi_at_xt(self, x, t):

        # get time index
        l = self.get_time_index(t)

        # get index of x
        k = self.sde.get_index(x)

        # evaluate psi at (x, t)
        return self.psi[l, k] if hasattr(self, 'psi') else None

    def get_value_function_at_xt(self, x, t):

        # get time index
        l = self.get_time_index(t)

        # get index of x
        k = self.sde.get_index(x)

        # evaluate value function at (x, t)
        return self.value_function[l, k] if hasattr(self, 'value_function') else None

    def get_u_opt_at_xt(self, x, t):

        # get time index
        l = self.get_time_index(t)

        # get index of x
        k = self.sde.get_index(x)

        # evaluate optimal control at (x, t)
        return self.u_opt[l, k] if hasattr(self, 'u_opt') else None

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

    def write_report(self, x, t):
        ''' writes the hjb solver parameters and the value of the solution at (x, t)

        Parameters
        ----------
        x: array
            point in the domain
        t: array
           time

        '''
        from sde_hjb_solver.utils import get_time_in_hms

        # time and space discretization
        print('\n space discretization')
        print('h = {:2.4f}'.format(self.sde.h))
        print('N_h = {:d}'.format(self.sde.Nh))
        print('\n time discretization')
        print('T = {:2.4f}'.format(self.T))
        print('dt = {:2.4f}'.format(self.dt))
        print('K = {:d}'.format(self.K))

        # psi, value function and control
        print('\n psi, value function and optimal control at (x, t)')

        print('x: {:2.3f}'.format(x))
        print('t = {:2.4f}'.format(t))
        psi = self.get_psi_at_xt(x, t)
        value_f = self.get_value_function_at_xt(x, t)
        u_opt = self.get_u_opt_at_xt(x, t)

        print('psi(x, t) = {:2.3e}\n'.format(psi))
        print('value_function(x, t) = {:2.3e}\n'.format(value_f))
        print('u_opt(x): {:2.3f}'.format(u_opt))

        # computational time
        h, m, s = get_time_in_hms(self.ct)
        print('\nComputational time: {:d}:{:02d}:{:02.2f}\n'.format(h, m, s))


    def plot_1d_psi_at_t(self, t0=0., t1=None, ylim=None):

        # final time as default value
        if t1 is None:
            t1 = self.T

        # get time indices
        l0 = self.get_time_index(t0)
        l1 = self.get_time_index(t1)

        # labels
        label1 = 't = {:2.2f})'.format(t0)
        label2 = 't = {:2.2f})'.format(t1)

        fig, ax = plt.subplots()
        ax.set_title(r'Estimation of $\Psi(x)$')
        ax.set_xlabel('x')
        ax.set_xlim(self.sde.domain)
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.plot(self.sde.domain_h, self.psi[l0, :], label=label1)
        ax.plot(self.sde.domain_h, self.psi[l1, :], label=label2)
        ax.legend()
        plt.show()

    def plot_1d_psi(self):

        fig, ax = plt.subplots()
        ax.set_title(r'Estimation of $\Psi(x)$')
        ax.set_xlabel('x')
        ax.set_ylabel('t')
        ax.set_xlim(self.sde.domain)

        im = ax.imshow(
            self.psi,
            vmin=0,
            vmax=1,
            origin='lower',
            extent=(self.sde.domain[0], self.sde.domain[1], 0, self.T),
            #cmap=cm.plasma,
            aspect='auto',
        )

        # add space for colour bar
        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
        fig.colorbar(im, cax=cbar_ax)

        plt.show()

    def plot_1d_value_function_at_t(self, t0=0, t1=None, ylim=None):
        # final time as default value
        if t1 is None:
            t1 = self.T

        # get time indices
        l0 = self.get_time_index(t0)
        l1 = self.get_time_index(t1)

        # labels
        label1 = 't = {:2.2f})'.format(t0)
        label2 = 't = {:2.2f})'.format(t1)

        fig, ax = plt.subplots()
        ax.set_title(r'Estimation of $\Phi(x, t)$')
        ax.set_xlabel('x')
        ax.set_xlim(self.sde.domain)
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.plot(self.sde.domain_h, self.value_function[l0, :], label=label1)
        ax.plot(self.sde.domain_h, self.value_function[l1, :], label=label2)
        ax.legend()
        plt.show()

    def plot_1d_perturbed_potential_at_t(self, t0=0, t1=None, ylim=None):

        # final time as default value
        if t1 is None:
            t1 = self.T

        # get time indices
        l0 = self.get_time_index(t0)
        l1 = self.get_time_index(t1)

        # labels
        label1 = 't = {:2.2f})'.format(t0)
        label2 = 't = {:2.2f})'.format(t1)

        fig, ax = plt.subplots()
        ax.set_title(r'Perturbed potential $(V + V_{bias})(x, t)$')
        ax.set_xlabel('x')
        ax.set_xlim(self.sde.domain)
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.plot(self.sde.domain_h, self.V)
        ax.plot(self.sde.domain_h, self.perturbed_potential[l0, :], label=label1)
        ax.plot(self.sde.domain_h, self.perturbed_potential[l1, :], label=label2)
        ax.legend()
        plt.show()

    def plot_1d_control_at_t(self, t0=0., t1=None, ylim=None):

        # final time as default value
        if t1 is None:
            t1 = self.T

        # get time indices
        l0 = self.get_time_index(t0)
        l1 = self.get_time_index(t1)

        # labels
        label1 = 't = {:2.2f})'.format(t0)
        label2 = 't = {:2.2f})'.format(t1)

        fig, ax = plt.subplots()
        ax.set_title(r'Optimal control $u^*(x, t)$')
        ax.set_xlabel('x')
        ax.set_xlim(self.sde.domain)
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.plot(self.sde.domain_h, self.u_opt[l0, :], label=label1)
        ax.plot(self.sde.domain_h, self.u_opt[l1, :], label=label2)
        ax.legend()
        plt.show()

    def plot_1d_control(self, vmin=None, vmax=None):

        fig, ax = plt.subplots()
        ax.set_title(r'Optimal control $u^*(x, t)$')
        ax.set_xlabel('x')
        ax.set_ylabel('t')
        ax.set_xlim(self.sde.domain)

        im = ax.imshow(
            self.u_opt,
            vmin=vmin,
            vmax=vmax,
            origin='lower',
            extent=(self.sde.domain[0], self.sde.domain[1], 0, self.T),
            #cmap=cm.plasma,
            aspect='auto',
        )

        # add space for colour bar
        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
        fig.colorbar(im, cax=cbar_ax)

        plt.show()

    def plot_1d_perturbed_drift_at_t(self, ylim=None):
        pass

import os
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg

from sde_hjb_solver.utils_path import save_data, load_data

class SolverHJB2D(object):
    ''' This class provides a solver of the following 2d BVP by using a
        finite differences method:
            0 = LΨ − f Ψ in S
            Ψ = exp(− g) in ∂S,
        where f and g are the running and terminal costs of the work functional,
        the solution Ψ is the quantity we want to estimate and L is the infinitessimal
        generator of the not controlled 2d diffusion process:
            L = b(x)·∇ + 1/2 sigma(x)^2·Δ

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


    Methods
    -------
    __init__(sde, h, load)

    start_timer()

    stop_timer()

    get_flatten_index(idx)

    get_bumpy_index(idx)

    get_x(k)

    get_flatten_idx_from_axis_neighbours(idx, i)

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

    plot_2d_psi(ylim=None)

    plot_2d_value_function(ylim=None)

    plot_2d_perturbed_potential(ylim=None)

    plot_2d_control(ylim=None)

    plot_2d_perturbed_drift(ylim=None)

    '''

    def __init__(self, sde, h, load=False):
        ''' init method

        Parameters
        ----------
        sde: langevinSDE object
            overdamped langevin sde object
        h: float
            step size
        load: bool
            load solution

        Raises
        ------
        NotImplementedError
            If dimension d is greater than 1
        '''

        if sde.d != 2:
            raise NotImplementedError('d > 2 not supported')

        # sde object
        self.sde = sde

        # discretization step
        self.h = h

        # rel directory path
        self.rel_dir_path = os.path.join(sde.name, 'h{:.0e}'.format(h))

        if load:
            self.load()

    def start_timer(self):
        ''' start timer
        '''
        self.ct_initial = time.perf_counter()

    def stop_timer(self):
        ''' stop timer
        '''
        self.ct_final = time.perf_counter()
        self.ct = self.ct_final - self.ct_initial

    def get_flatten_index(self, idx):
        ''' maps the bumpy index of the node (index of each axis) to
            the flatten index of the node, i.e. the node number.

        Parameters
        ----------
        idx: tuple
            bumpy index of the node

        Returns
        -------
        int
            flatten index of the node
        '''
        assert type(idx) == tuple, ''
        assert len(idx) == self.sde.d, ''

        k = 0
        for i in range(self.sde.d):
            assert 0 <= idx[i] <= self.sde.Nx[i] - 1, ''
            Nx_prod = 1
            for j in range(i+1, self.sde.d):
                Nx_prod *= self.sde.Nx[j]
            k += idx[i] * Nx_prod

        return k

    def get_bumpy_index(self, k):
        ''' maps the flatten index of the node (node number) to
            the bumpy index of the node.

        Parameters
        ----------
        k: int
            flatten index of the node

        Returns
        -------
        tuple
            bumpy index of the node
        '''
        #assert type(k) == int, ''
        assert 0 <= k <= self.sde.Nh -1, ''

        idx = [None for i in range(self.sde.d)]
        for i in range(self.sde.d):
            Nx_prod = 1
            for j in range(i+1, self.sde.d):
                Nx_prod *= self.sde.Nx[j]
            idx[i] = k // Nx_prod
            k -= idx[i] * Nx_prod
        return tuple(idx)

    def get_x(self, k):
        ''' returns the x-coordinate of the node k

        Parameters
        ----------
        k: int
            flatten index of the node

        Returns
        -------
        float
            point in the domain
        '''
        assert k in np.arange(self.sde.Nh), ''

        return self.sde.domain_h.reshape(self.sde.Nh, self.sde.d)[k]

    def get_flatten_idx_from_axis_neighbours(self, idx, i):
        ''' get flatten idx of the neighbours with respect to the i-th coordinate

        Parameters
        ----------
        idx: tuple
            bumpy index of the node
        i: int
            index of the ith coordinate

        Returns
        -------
        tuple
            (k_left, k_right)
        '''

        # find flatten index of left neighbour wrt the i axis
        if idx[i] == 0:
            k_left = None
        else:
            left_idx = list(idx)
            left_idx[i] = idx[i] - 1
            k_left = self.get_flatten_index(tuple(left_idx))

        # find flatten index of right neighbour wrt the i axis
        if idx[i] == self.sde.Nx[i] - 1:
            k_right = None
        else:
            right_idx = list(idx)
            right_idx[i] = idx[i] + 1
            k_right = self.get_flatten_index(tuple(right_idx))

        return (k_left, k_right)

    def solve_bvp(self):
        ''' solves bvp by using finite difference
        '''

        # start timer
        self.start_timer()

        # discretized step
        h = self.h

        # discretize domain
        self.sde.discretize_domain_2d(h)

        # assemble linear system of equations: A \Psi = b.
        A = sparse.lil_matrix((self.sde.Nh, self.sde.Nh))
        b = np.zeros(self.sde.Nh)

        for k in np.arange(self.sde.Nh):


            # get discretized domain index
            idx = self.get_bumpy_index(k)

            # get point
            #x = self.get_x(idx)
            x = self.get_x(k)

            # assemble matrix A and vector b on S
            if k not in self.sde.ts_idx and k not in self.sde.boundary_idx:

                # drift and diffusion at x
                drift = self.sde.drift(x)
                sigma = self.sde.diffusion

                A[k, k] = - (sigma**2 * self.sde.d) / h**2 - self.sde.f(x)

                # x-axis neighbours
                k_left, k_right = self.get_flatten_idx_from_axis_neighbours(idx, i=0)
                A[k, k_left] = sigma**2 / (2 * h**2) - drift[0] / (2 * h)
                A[k, k_right] = sigma**2 / (2 * h**2) + drift[0] / (2 * h)

                # y-axis neighbours
                k_left, k_right = self.get_flatten_idx_from_axis_neighbours(idx, i=1)
                A[k, k_left] = sigma**2 / (2 * h**2) - drift[1] / (2 * h)
                A[k, k_right] = sigma**2 / (2 * h**2) + drift[1] / (2 * h)

            # impose condition on ∂S
            elif k in self.sde.ts_idx and not k in self.sde.boundary_idx:
                A[k, k] = 1
                b[k] = np.exp(- self.sde.g(x))

            # stability condition on the boundary: Psi should be flat
            elif k in self.sde.boundary_idx:
                neighbour_counter = 0

                if k in self.sde.boundary_x_idx:

                    # add neighbour
                    k_left, k_right = self.get_flatten_idx_from_axis_neighbours(idx, i=0)
                    if k_left is not None:
                        A[k, k_left] = - 1
                    elif k_right is not None:
                        A[k, k_right] = - 1

                    # update counter
                    neighbour_counter += 1

                if k in self.sde.boundary_y_idx:

                    # add neighbour
                    k_left, k_right = self.get_flatten_idx_from_axis_neighbours(idx, i=1)
                    if k_left is not None:
                        A[k, k_left] = - 1
                    elif k_right is not None:
                        A[k, k_right] = - 1

                    # update counter
                    neighbour_counter += 1

                # normalize
                A[k, k] = neighbour_counter

        # solve linear system and save
        psi = linalg.spsolve(A.tocsc(), b)
        self.psi = psi.reshape(self.sde.Nx)
        self.solved = True

        # stop timer
        self.stop_timer()

    def compute_value_function(self):
        ''' computes the value function
                phi = - log (psi)
        '''
        self.value_function =  - np.log(self.psi)

    def compute_optimal_control(self):
        ''' computes by finite differences the optimal control
                u_opt = - sigma ∇ value_f
        '''
        assert hasattr(self, 'value_function'), ''
        assert self.value_function.ndim == self.sde.d, ''

        # diffusion term
        sigma = self.sde.diffusion

        # preallocate optimal control
        self.u_opt = np.zeros(self.sde.Nx + (self.sde.d, ))

        for i in range(self.sde.d):

            # idx value f type
            value_f_k_plus_idx = [slice(self.sde.Nx[0]), slice(self.sde.Nx[1])]
            value_f_k_minus_idx = [slice(self.sde.Nx[0]), slice(self.sde.Nx[1])]

            # idx u type
            u_k_idx = [slice(self.sde.Nx[0]), slice(self.sde.Nx[1]), i]
            u_0_idx = [slice(self.sde.Nx[0]), slice(self.sde.Nx[1]), i]
            u_1_idx = [slice(self.sde.Nx[0]), slice(self.sde.Nx[1]), i]
            u_N_minus_idx = [slice(self.sde.Nx[0]), slice(self.sde.Nx[1]), i]
            u_N_idx = [slice(self.sde.Nx[0]), slice(self.sde.Nx[1]), i]

            value_f_k_plus_idx[i] = slice(2, self.sde.Nx[i])
            value_f_k_minus_idx[i] = slice(0, self.sde.Nx[i] - 2)
            u_k_idx[i] = slice(1, self.sde.Nx[i] - 1)
            u_0_idx[i] = 0
            u_1_idx[i] = 1
            u_N_minus_idx[i] = self.sde.Nx[i] - 2
            u_N_idx[i] = self.sde.Nx[i] - 1

            # generalized central difference
            self.u_opt[tuple(u_k_idx)] = - sigma *(
                self.value_function[tuple(value_f_k_plus_idx)]
              - self.value_function[tuple(value_f_k_minus_idx)]
            ) / (2 * self.sde.h)
            self.u_opt[tuple(u_0_idx)] = self.u_opt[tuple(u_1_idx)]
            self.u_opt[tuple(u_N_idx)] = self.u_opt[tuple(u_N_minus_idx)]


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
        idx = self.sde.get_idx(x)

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
        idx = self.sde.get_idx(x)

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
        idx = self.sde.get_idx(x)

        # evaluate optimal control at x
        return self.u_opt[idx] if hasattr(self, 'u_opt') else None

    def get_perturbed_potential_and_drift(self):
        ''' computes the potential, bias potential, controlled potential, gradient,
            controlled drift
        '''

        # flatten domain_h
        x = self.sde.domain_h.reshape(self.sde.Nh, self.sde.d)

        # diffusion term
        sigma = self.sde.diffusion

        # potential, bias potential and tilted potential
        self.V = self.sde.potential(x).reshape(self.sde.Nx)
        self.bias_potential = (sigma**2) * self.value_function
        self.perturbed_potential = self.V + self.bias_potential

        # gradient and tilted drift
        self.dV = self.sde.gradient(x).reshape(self.sde.domain_h.shape)
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

    def plot_2d_psi(self):
        fig, ax = plt.subplots()
        ax.set_title(r'Estimation of $\Psi(x)$')
        ax.set_xlabel(r'$x_1$')
        ax.set_ylabel(r'$x_2$')
        ax.set_xlim(self.sde.domain[0])
        ax.set_ylim(self.sde.domain[1])

        # contour f
        cs = ax.contourf(
            self.sde.domain_h[:, :, 0],
            self.sde.domain_h[:, :, 1],
            self.psi,
            extend='both',
        )

        # colorbar
        cbar = fig.colorbar(cs)

        plt.show()

    def plot_2d_value_function(self):
        fig, ax = plt.subplots()
        ax.set_title(r'Estimation of $\Phi(x)$')
        ax.set_xlabel(r'$x_1$')
        ax.set_ylabel(r'$x_2$')
        ax.set_xlim(self.sde.domain[0])
        ax.set_ylim(self.sde.domain[1])

        # contour f
        cs = ax.contourf(
            self.sde.domain_h[:, :, 0],
            self.sde.domain_h[:, :, 1],
            self.value_function,
            extend='both',
        )

        # colorbar
        cbar = fig.colorbar(cs)

        plt.show()

    def plot_2d_perturbed_potential(self):
        fig, ax = plt.subplots()
        ax.set_title(r'Perturbed potential $(V + V_{bias})(x)$')
        ax.set_xlabel(r'$x_1$')
        ax.set_ylabel(r'$x_2$')
        ax.set_xlim(self.sde.domain[0])
        ax.set_ylim(self.sde.domain[1])

        # contour f
        cs = ax.contourf(
            self.sde.domain_h[:, :, 0],
            self.sde.domain_h[:, :, 1],
            self.perturbed_potential,
            extend='both',
        )

        # colorbar
        cbar = fig.colorbar(cs)

        plt.show()

    def plot_2d_control(self, scale=None, width=0.005):
        from matplotlib import colors, cm

        fig, ax = plt.subplots()
        ax.set_title(r'Optimal control $u^*(x)$')
        ax.set_xlabel(r'$x_1$')
        ax.set_ylabel(r'$x_2$')
        ax.set_xlim(self.sde.domain[0])
        ax.set_ylim(self.sde.domain[1])

        X = self.sde.domain_h[:, :, 0]
        Y = self.sde.domain_h[:, :, 1]
        U = self.u_opt[:, :, 0]
        V = self.u_opt[:, :, 1]

        # set colormap
        colormap = cm.get_cmap('viridis_r', 100)
        colormap = colors.ListedColormap(
            colormap(np.linspace(0.20, 0.95, 75))
        )

        # initialize norm object and make rgba array
        C = np.sqrt(U**2 + V**2)
        norm = colors.Normalize(vmin=np.min(C), vmax=np.max(C))
        sm = cm.ScalarMappable(cmap=colormap, norm=norm)

        # quiver
        quiv = ax.quiver(
            X,
            Y,
            U,
            V,
            C,
            cmap=colormap,
            angles='xy',
            scale_units='xy',
            scale=scale,
            width=width,
        )

        # colorbar
        fig.colorbar(sm)

        plt.show()


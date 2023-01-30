import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg

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
    mfht: array
       mean first hitting time


    Methods
    -------
    __init__(sde, h)

    start_timer()

    stop_timer()

    get_flatten_index(idx)

    get_bumpy_index(idx)

    get_x(k)

    get_flatten_idx_from_axis_neighbours(idx, i)

    solve_bvp()

    compute_value_function()

    compute_optimal_control()

    compute_mfht()

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

        if sde.d != 2:
            raise NotImplementedError('d > 2 not supported')

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
            idx_left = list(idx)
            idx_left[i] = idx[i] - 1
            k_left = self.get_flatten_index(tuple(idx_left))

        # find flatten index of right neighbour wrt the i axis
        if idx[i] == self.sde.Nx[i] - 1:
            k_right = None
        else:
            idx_right = list(idx)
            idx_right[i] = idx[i] + 1
            k_right = self.get_flatten_index(tuple(idx_right))

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
            if k not in self.sde.idx_ts and k not in self.sde.idx_boundary:

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
            elif k in self.sde.idx_ts and not k in self.sde.idx_boundary:
                A[k, k] = 1
                b[k] = np.exp(- self.sde.g(x))

            # stability condition on the boundary: Psi should be flat
            elif k in self.sde.idx_boundary:
                neighbour_counter = 0

                if k in self.sde.idx_boundary_x:

                    # add neighbour
                    k_left, k_right = self.get_flatten_idx_from_axis_neighbours(idx, i=0)
                    if k_left is not None:
                        A[k, k_left] = - 1
                    elif k_right is not None:
                        A[k, k_right] = - 1

                    # update counter
                    neighbour_counter += 1

                if k in self.sde.idx_boundary_y:

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
            idx_value_f_k_plus = [slice(self.sde.Nx[0]), slice(self.sde.Nx[1])]
            idx_value_f_k_minus = [slice(self.sde.Nx[0]), slice(self.sde.Nx[1])]

            # idx u type
            idx_u_k = [slice(self.sde.Nx[0]), slice(self.sde.Nx[1]), i]
            idx_u_0 = [slice(self.sde.Nx[0]), slice(self.sde.Nx[1]), i]
            idx_u_1 = [slice(self.sde.Nx[0]), slice(self.sde.Nx[1]), i]
            idx_u_N_minus = [slice(self.sde.Nx[0]), slice(self.sde.Nx[1]), i]
            idx_u_N = [slice(self.sde.Nx[0]), slice(self.sde.Nx[1]), i]

            idx_value_f_k_plus[i] = slice(2, self.sde.Nx[i])
            idx_value_f_k_minus[i] = slice(0, self.sde.Nx[i] - 2)
            idx_u_k[i] = slice(1, self.sde.Nx[i] - 1)
            idx_u_0[i] = 0
            idx_u_1[i] = 1
            idx_u_N_minus[i] = self.sde.Nx[i] - 2
            idx_u_N[i] = self.sde.Nx[i] - 1

            # generalized central difference
            self.u_opt[tuple(idx_u_k)] = - sigma *(
                self.value_function[tuple(idx_value_f_k_plus)]
              - self.value_function[tuple(idx_value_f_k_minus)]
            ) / (2 * self.sde.h)
            self.u_opt[tuple(idx_u_0)] = self.u_opt[tuple(idx_u_1)]
            self.u_opt[tuple(idx_u_N)] = self.u_opt[tuple(idx_u_N_minus)]


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
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)

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
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)

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
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)

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
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)

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

    """
    def plot_2d_perturbed_drift(self):
        fig, ax = plt.subplots()
        ax.set_title(r'Perturbed drift $\nabla(V + V_{bias})(x)$')
        self.get_perturbed_potential_and_drift()
        plt.show()
    """

    def plot_1d_mfht(self, ylim=None):
        fig, ax = plt.subplots()
        ax.set_title(r'Estimation of $\mathbb{E}^x[\tau]$')
        ax.set_xlabel('x')
        ax.set_xlim(self.sde.domain)
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.plot(self.sde.domain_h, self.mfht)
        plt.show()
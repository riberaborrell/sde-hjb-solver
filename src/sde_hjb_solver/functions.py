import functools

import numpy as np

def constant(x, a):
    ''' constant scalar function
    '''

    # convertion
    x = np.asarray(x)
    a = float(a)

    # scalar input
    if x.ndim == 0:
        return a

    # array input
    elif x.ndim == 1:
        return np.array([a], dtype=np.float32)

    # batch input
    elif x.ndim == 2:
        K = x.shape[0]
        return a * np.ones(K)

def linear(x, a):
    ''' linear scalar function
    '''

    # convertion
    x = np.asarray(x)

    # scalar input
    if x.ndim == 0:
        return a * x

    # array input
    elif x.ndim == 1:
        return np.dot(a, x)

    # batch input
    elif x.ndim == 2:
        K = x.shape[0]
        return (a * x).squeeze()


def quadratic_one_well(x, nu):
    ''' Multi-dimensional quadratic one well. V(x): \R^d -> \R.
        For d=1 V(x; nu) = nu * (x -1)**2 and has a minimum at x=1.
    '''

    # convertion
    x = np.asarray(x)
    nu = np.asarray(nu)
    if nu.ndim == 0:
        nu = nu[np.newaxis]

    # scalar input
    if x.ndim == 0:
        return nu[0] * (x - 1) ** 2

    assert nu.ndim == 1, ''
    d = nu.shape[0]

    # array input
    if x.ndim == 1:
        assert x.shape[0] == d, ''
        return np.sum(nu * (x -1)**2)

    # batch input
    elif x.ndim == 2:
        assert x.shape[1] == d, ''
        return np.sum(nu * (x -1)**2, axis=1)

def double_well(x, alpha):
    ''' Multi-dimensional double well. V(x): \R^d -> \R.
        For d=1 V has minimums at x=+- 1 and maximum at x=alpha.
    '''

    # convertion
    x = np.asarray(x)
    alpha = np.asarray(alpha)
    if alpha.ndim == 0:
        alpha = alpha[np.newaxis]

    # scalar input
    if x.ndim == 0:
        return alpha[0] * (x**2 - 1) ** 2

    assert alpha.ndim == 1, ''
    d = alpha.shape[0]

    # array input
    if x.ndim == 1:
        assert x.shape[0] == d, ''
        return np.sum(alpha * (x**2 - 1) ** 2)

    # batch input
    elif x.ndim == 2:
        assert x.shape[1] == d, ''
        return np.sum(alpha * (x ** 2 -1) **2, axis=1)

def double_well_gradient(x, alpha):
    ''' Gradient of the multi-dimensional double well. ∇V(x): \R^d -> \R^d.
    '''

    # convertion
    x = np.asarray(x)
    alpha = np.asarray(alpha)
    if alpha.ndim == 0:
        alpha = alpha[np.newaxis]

    # scalar input
    if x.ndim == 0:
        return 4 * alpha[0] * x * (x**2 - 1)

    assert alpha.ndim == 1, ''
    d = alpha.shape[0]

    # array input
    if x.ndim == 1:
        assert x.shape[0] == d, ''
        return 4 * alpha * x * (x**2 - 1)

    # batch input
    elif x.ndim == 2:
        assert x.shape[1] == d, ''
        K = x.shape[0]
        return 4 * alpha * x * (x ** 2 - 1)

def skew_double_well_1d(x):
    ''' Skew 1-dimensional double well.
    '''
    d = 1
    x = np.asarray(x)

    # array input
    if x.ndim == 1:
        assert x.shape[0] == d, ''

    # batch input
    elif x.ndim == 2:
        assert x.shape[1] == d, ''

    return (x**2 -1)**2 - 0.2*x + 0.3

def skew_double_well_gradient_1d(x):
    ''' Gradient of the skew 1-dimensional double well.
    '''
    x = np.asarray(x)

    # array input
    if x.ndim == 1:
        assert x.shape[0] == 1, ''

    # batch input
    elif x.ndim == 2:
        assert x.shape[1] == 1, ''

    return 4 * x * (x**2 - 1) - 0.2


def double_well_curved_2d(x):
    '''
    '''
    d = 2

    # scalar input
    if x.ndim == 0:
        raise ValueError('The input needs to be of array type')

    # array input
    elif x.ndim == 1:
        assert x.shape[0] == d, ''
        x = np.expand_dims(x, axis=0)
        is_array_input = True

    # batch input
    elif x.ndim == 2:
        assert x.shape[1] == d, ''
        is_array_input = False

    potential = (x[:, 0]**2 - 1) ** 2 + 2 *(x[:, 0]**2 + x[:, 1] - 1) ** 2

    if is_array_input:
        potential = potential.squeeze()

    return potential

def double_well_curved_gradient_2d(x):
    '''
    '''
    d = 2

    # scalar input
    if x.ndim == 0:
        raise ValueError('The input needs to be of array type')

    # array input
    elif x.ndim == 1:
        assert x.shape[0] == d, ''
        x = np.expand_dims(x, axis=0)
        is_array_input = True

    # batch input
    elif x.ndim == 2:
        assert x.shape[1] == d, ''
        is_array_input = False

    partial_x = 4 * x[:, 0] * (x[:, 0]**2 - 1) + 8 * x[:, 1] *(x[:, 0]**2 + x[:, 1] - 1)
    partial_y = 4 * x[:, 1] * (x[:, 0]**2 + x[:, 1] - 1)
    gradient = np.hstack((partial_x, partial_y))

    if is_array_input:
        gradient = gradient.squeeze()

    return gradient

def triple_well_2d(x, alpha):
    '''
    '''
    d = 2

    # scalar input
    if x.ndim == 0:
        raise ValueError('The input needs to be of array type')

    # array input
    elif x.ndim == 1:
        assert x.shape[0] == d, ''
        x = np.expand_dims(x, axis=0)
        is_array_input = True

    # batch input
    elif x.ndim == 2:
        assert x.shape[1] == d, ''
        is_array_input = False

    potential = alpha * (
        + 3 * np.exp(- x[:, 0]**2 - (x[:, 1] - (1./ 3))**2)
        - 3 * np.exp(- x[:, 0]**2 - (x[:, 1] -(5./3))**2)
        - 5 * np.exp(- (x[:, 0] - 1)**2 - x[:, 1]**2)
        - 5 * np.exp(- (x[:, 0] + 1)**2 - x[:, 1]**2)
        + 0.2 * (x[:, 0]**4)
        + 0.2 * (x[:, 1] -(1./3))**4
    )

    if is_array_input:
        potential = potential.squeeze()

    return potential

def triple_well_gradient_2d(x, alpha):
    '''
    '''
    d = 2

    # scalar input
    if x.ndim == 0:
        raise ValueError('The input needs to be of array type')

    # array input
    elif x.ndim == 1:
        assert x.shape[0] == d, ''
        x = np.expand_dims(x, axis=0)
        is_array_input = True

    # batch input
    elif x.ndim == 2:
        assert x.shape[1] == d, ''
        is_array_input = False

    partial_x = alpha * (
        - 6 * x[:, 0] * np.exp(- x[:, 0]**2 - (x[:, 1] - (1./ 3))**2)
        - 6 * x[:, 0] * np.exp(- x[:, 0]**2 - (x[:, 1] - (5./3))**2)
        + 10 * (x[:, 0] - 1) * np.exp(- (x[:, 0] - 1)**2 - x[:, 1]**2)
        + 10 * (x[:, 0] + 1) * np.exp(- (x[:, 0] + 1)**2 - x[:, 1]**2)
        + (4 / 5) * (x[:, 0]**3)
    )

    partial_y = alpha * (
        - 6 * (x[:, 1] - (1./3)) * np.exp(- x[:, 0]**2 - (x[:, 1] - (1./ 3))**2)
        - 6 * (x[:, 1] - (5./3)) * np.exp(- x[:, 0]**2 - (x[:, 1] - (5./3))**2)
        + 10 * x[:, 1] * np.exp(- (x[:, 0] - 1)**2 - x[:, 1]**2)
        + 10 * x[:, 1] * np.exp(- (x[:, 0] + 1)**2 - x[:, 1]**2)
        + (4 / 5) * (x[:, 1] - (1./3))**3
    )

    gradient = np.hstack((partial_x, partial_y))

    if is_array_input:
        gradient = gradient.squeeze()

    return gradient

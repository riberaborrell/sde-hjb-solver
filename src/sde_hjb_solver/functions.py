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

def triple_well_1d(x):
    ''' Asymmetric 1-dimensional triple well
    '''
    d = 1
    x = np.asarray(x)

    # array input
    if x.ndim == 1:
        assert x.shape[0] == d, ''

    # batch input
    elif x.ndim == 2:
        assert x.shape[1] == d, ''

    return (0.5 * x**6 - 15 * x**4 + 119 * x**2 + 28*x + 50) / 200

def triple_well_gradient_1d(x):
    ''' Gradient of the asymmetric 1-dimensional triple well.
    '''
    x = np.asarray(x)

    # array input
    if x.ndim == 1:
        assert x.shape[0] == 1, ''

    # batch input
    elif x.ndim == 2:
        assert x.shape[1] == 1, ''

    return + (3 * x**5 - 60 * x**3 + 238 * x + 28) / 200 \

def ryck_bell_1d(x):
    return np.polyval(np.array([-3.778, 3.156, -0.368, -1.578, 1.462, 1.116]), np.cos(x))

def ryck_bell_gradient_1d(x):
    return np.polyval(np.array([18.94, -12.624, 1.104, 3.156, -1.462]), np.cos(x)) * np.sin(x)

def five_well_1d(x):
    ''' 1-dimensional five well potential.
    '''
    d = 1
    x = np.asarray(x)

    # array input
    if x.ndim == 1:
        assert x.shape[0] == d, ''

    # batch input
    elif x.ndim == 2:
        assert x.shape[1] == d, ''

    return + (0.5 * x**6 - 15 * x**4 + 119 * x**2 + 28*x + 50) / 200 \
           - 0.6 * np.exp(-12.5 * (x + 2)**2 ) \
           - 0.7 * np.exp(-12.5 * (x - 1.8)**2)

def five_well_gradient_1d(x):
    ''' Gradient of the 1-dimensional five well potential.
    '''
    x = np.asarray(x)

    # array input
    if x.ndim == 1:
        assert x.shape[0] == 1, ''

    # batch input
    elif x.ndim == 2:
        assert x.shape[1] == 1, ''

    return + (3 * x**5 - 60 * x**3 + 238 * x + 28) / 200 \
           - 15 * (x + 2) * np.exp(-12.5 *(x + 2)**2) \
           - 17.5 * (x - 1.8) * np.exp(-12.5*(x - 1.8)**2)

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
        - 3 * np.exp(- x[:, 0]**2 - (x[:, 1] - (5./3))**2)
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
        + 6 * x[:, 0] * np.exp(- x[:, 0]**2 - (x[:, 1] - (5./3))**2)
        + 10 * (x[:, 0] - 1) * np.exp(- (x[:, 0] - 1)**2 - x[:, 1]**2)
        + 10 * (x[:, 0] + 1) * np.exp(- (x[:, 0] + 1)**2 - x[:, 1]**2)
        + (4 / 5) * (x[:, 0]**3)
    )

    partial_y = alpha * (
        - 6 * (x[:, 1] - (1./3)) * np.exp(- x[:, 0]**2 - (x[:, 1] - (1./ 3))**2)
        + 6 * (x[:, 1] - (5./3)) * np.exp(- x[:, 0]**2 - (x[:, 1] - (5./3))**2)
        + 10 * x[:, 1] * np.exp(- (x[:, 0] - 1)**2 - x[:, 1]**2)
        + 10 * x[:, 1] * np.exp(- (x[:, 0] + 1)**2 - x[:, 1]**2)
        + (4 / 5) * (x[:, 1] - (1./3))**3
    )

    gradient = np.hstack((partial_x, partial_y))

    if is_array_input:
        gradient = gradient.squeeze()

    return gradient

def mueller_brown_2d(x):
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

    A = [-200, -100, -170, 15]
    a = [-1, -1, -6.5, 0.7]
    b = [0, 0, 11, 0.6]
    c = [-10, -10, -6.5, 0.7]
    x1_hat = [1, 0, -0.5, -1]
    x2_hat = [0, 0.5, 1.5, 1]

    K = x.shape[0]
    potential = np.zeros(K)
    for i in range(0, 4):
        potential += A[i] * np.exp(
            a[i] * (x[:, 0] - x1_hat[i])**2
            + b[i] * (x[:, 0] - x1_hat[i]) * (x[:, 1] - x2_hat[i])
            + c[i] * (x[:, 1] - x2_hat[i])**2)

    if is_array_input:
        potential = potential.squeeze()

    return potential

def mueller_brown_gradient_2d(x):
    '''
    '''
    d = 2

    A = [-200, -100, -170, 15]
    a = [-1, -1, -6.5, 0.7]
    b = [0, 0, 11, 0.6]
    c = [-10, -10, -6.5, 0.7]
    x1_hat = [1, 0, -0.5, -1]
    x2_hat = [0, 0.5, 1.5, 1]

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

    K = x.shape[0]
    partial_x = np.zeros(K)
    partial_y = np.zeros(K)

    for i in range(0, 4):
        exp_term = np.exp(
            a[i] * (x[:, 0] - x1_hat[i])**2
            + b[i] * (x[:, 0] - x1_hat[i]) * (x[:, 1] - x2_hat[i])
            + c[i] * (x[:, 1] - x2_hat[i])**2)
        partial_x += A[i] * (2 * a[i] * (x[:, 0] - x1_hat[i]) + b[i] * (x[:, 1] - x2_hat[i])) * exp_term
        partial_y += A[i] * (b[i] * (x[:, 0] - x1_hat[i]) + 2 * c[i] * (x[:, 1] - x2_hat[i])) * exp_term

    gradient = np.hstack((partial_x, partial_y))

    if is_array_input:
        gradient = gradient.squeeze()

    return gradient

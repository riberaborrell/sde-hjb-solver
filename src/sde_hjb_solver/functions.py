import functools

import numpy as np

FLOAT_TYPES = [float, np.float32, np.float64]

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

def quadratic_one_well(x, nu):
    ''' Multi-dimensional quadratic one well. V(x): \R^d -> \R
        For d=1 V has a minimum at x=1.
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
    x = np.asarray(x)

    # array input 
    if x.ndim == 1:
        assert x.shape[0] == 1, ''

    # batch input 
    elif x.ndim == 2:
        assert x.shape[1] == 1, ''

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


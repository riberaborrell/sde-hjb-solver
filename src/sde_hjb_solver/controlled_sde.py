import functools

from sde_hjb_solver.functions import *

class ControlledSDE(object):
    '''
    '''

    def __init__(self, d, domain=None, **kwargs):

        # dimension
        self.d = d

        # domain bounds
        self.domain = domain

    def set_mgf_setting(self, lam=1.):
        ''' Set moment generating function of the first hitting time setting
        '''
        # set mgf problem flag
        self.setting = 'mgf'

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
        self.setting = 'committor'

        # running and final costs
        self.epsilon = epsilon
        self.f = lambda x: 0
        self.g = lambda x: np.where(
            self.is_target_set_b(x),
            -np.log(1+epsilon),
            -np.log(epsilon),
        )

        # target set indices
        self.get_target_set_idx = self.get_target_set_idx_committor

    def set_finite_time_horizon_setting(self, nu=1.0):
        ''' Set finite time horizon setting
        '''
        # set committor problem flag
        self.setting = 'finite_time_horizon'

        # running and final costs
        self.nu = nu
        self.f = lambda x: 0
        self.g = functools.partial(quadratic_one_well, nu=nu)

    def set_fht_probs_setting(self, T=1.0, epsilon=1e-10):
        ''' Set first hitting time probabilities setting
        '''
        # set committor problem flag
        self.setting = 'fht_probabilities'

        # finite time horizon
        self.T = T

        # running and final costs
        self.epsilon = epsilon
        self.f = lambda x: 0
        self.g = lambda x: np.where(
            self.is_target_set(x),
            -np.log(1+epsilon),
            -np.log(epsilon),
        )



    def __str__(self):
        return f'{self.name}__{self.params_str}'

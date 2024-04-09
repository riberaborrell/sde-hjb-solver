import functools

from sde_hjb_solver.functions import *

class ControlledSDE(object):
    '''
    '''

    def __init__(self, d, domain=None):

        # dimension
        self.d = d

        # domain bounds
        self.domain = domain

        # problem types flags
        self.overdamped_langevin = False

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

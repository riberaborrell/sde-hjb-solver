class OrnsteinUhlenbeckSDE1D(ControlledSDE1D):
    '''
    '''

    def __init__(self, theta=1., sigma=1., domain=None):
        super().__init__(domain=domain)

        assert theta > 0, ''
        assert sigma > 0, ''

        # parameters
        self.theta = theta
        self.sigma = sigma

        # drift term
        self.drift = lambda x: - theta * x

        # diffusion
        self.diffusion = sigma

class OrnsteinUhlenbeckStoppingTime1D(OrnsteinUhlenbeckSDE1D):
    '''
    '''

    def __init__(self, theta=1., sigma=1., lam=1.0, domain=None, target_set=None):
        super().__init__(theta=theta, sigma=sigma)

        # log name
        self.name = 'ornstein-uhlenbeck-1d-st__theta{:.1f}_sigma{:.1f}'.format(theta, sigma)

        # domain
        if self.domain is None:
            self.domain = (-2, 2)

        # target set
        if target_set is not None:
            self.target_set = target_set
        else:
            self.target_set = (1, 2)

        # stopping time setting
        self.set_stopping_time_setting(lam=lam)


# target set condition
        #self.is_in_target_set_condition = lambda x: (x >= self.target_set[:, 0]) & (x <= self.target_set[:, 1])

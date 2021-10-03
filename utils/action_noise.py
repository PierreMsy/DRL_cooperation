import numpy as np
from copy import copy


class NoiseCreator():

    def __init__(self):
        self.builders = {
            'OU': lambda size, kwargs : OUActionNoise(size, **kwargs)
        }

    def create(self, noise, size, kwargs):
        return self.builders[noise](size, kwargs)
        
class OUActionNoise:
    """
    Ornstein-Uhlenbeck process.
    Temporally correlated noise that has a zero mean used to add
    exploratioin to the deterministic policy.
    """
    def __init__(self,
                 size,
                 mu,
                 theta,
                 sigma):
        self.mu = mu  * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.state = np.zeros_like(self.mu)

    def sample(self):

        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.normal(size=self.mu.shape)
        x = self.state + dx

        self.state = x
        return x

    def reset(self):
        '''
        reset the temporal correlation.
        '''
        self.state = copy(self.mu)

    def __repr__(self):
        return f'OrnsteinUhlenbeckActionNoise(mu={self.mu}, sigma={self.sigma})'
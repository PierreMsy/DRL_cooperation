import numpy as np
from copy import copy


class NoiseCreator():

    def __init__(self):
        self.builders = {
            'OU': lambda size, kwargs : OUActionNoise(size, **kwargs),
            'scaling_OU': lambda size, kwargs : Scaling_OUActionNoise(size, **kwargs)
        }

    def create(self, noise, size, kwargs):
        return self.builders[noise](size, kwargs)

class Scaling_OUActionNoise:
    """
    Ornstein-Uhlenbeck process.
    Temporally correlated noise that has a zero mean used to add
    exploratioin to the deterministic policy.

    """
    def __init__(self,
                 size,
                 mu,
                 theta, theta_max, theta_grow,
                 sigma, sigma_min, sigma_decay):
        """
            mu : mean
            theta : attraction of the mean
            sigma : magnitude of the wiener steps
        """
        self.mu = mu  * np.ones(size)
        self.theta_init = theta
        self.theta = theta
        self.theta_max = theta_max
        self.theta_grow = theta_grow
        self.sigma_init = sigma
        self.sigma = sigma
        self.sigma_min = sigma_min
        self.sigma_decay = sigma_decay
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
    
    def update(self):
        self.sigma = max(self.sigma_min, self.sigma * self.sigma_decay)
        self.theta = min(self.theta_max, self.theta * self.theta_grow)

    def __repr__(self):
        representation =(
            f'OrnsteinUhlenbeckActionNoise(mu={self.mu}, sigma={self.sigma}, theta={self.theta})\n'
            + f'init values were sigma={self.sigma_init}, theta={self.theta_init}')
        return representation
        
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
        """
            mu : mean
            theta : attraction of the mean
            sigma : magnitude of the wiener steps
        """
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
    
    def update(self):
        pass

    def __repr__(self):
        return f'OrnsteinUhlenbeckActionNoise(mu={self.mu}, sigma={self.sigma})'
import numpy as np
from copy import copy


class NoiseCreator():
    '''
    Factory that build the action noise responsible for the exploration of the agent.
    '''
    def __init__(self):
        self.builders = {
            'OU': lambda size, kwargs : OUActionNoise(size, **kwargs),
            'scaling_OU': lambda size, kwargs : Scaling_OUActionNoise(size, **kwargs),
            'gaussian':  lambda size, kwargs : GaussianNoise(size, **kwargs)
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
        """
        mu : mean
        theta : attraction of the mean
        sigma : magnitude of the wiener steps
        """
        self.mu = mu  
        self.theta = theta
        self.sigma = sigma
        self.state = np.zeros(size)
        self.size = size

    def sample(self):
        '''
        sample a new point from the OU process, update the state and return a noise
        of the asked size.
        '''
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.normal()
        x = self.state + dx

        self.state = x
        return x

    def sample_multipe(self, noise_size):
        '''
        sample a new point from the OU process, update the state and return a noise
        of the asked size.
        '''
        dx = self.theta * (self.mu * np.ones(self.size) - self.state) + self.sigma * np.random.normal(size=self.state.shape)
        x = self.state + dx

        self.state = x
        return np.squeeze(np.tile(x, (noise_size, 1)))

    def reset(self):
        '''
        reset the temporal correlation.
        '''
        self.state = copy(self.mu)
    
    def update(self):
        pass

    def __repr__(self):
        return f'OrnsteinUhlenbeckActionNoise(mu={self.mu}, sigma={self.sigma})'

class Scaling_OUActionNoise(OUActionNoise):
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
        super().__init__(size, mu, theta, sigma)
        
        self.theta_init = theta
        self.theta_max = theta_max
        self.theta_grow = theta_grow

        self.sigma_init = sigma
        self.sigma_min = sigma_min
        self.sigma_decay = sigma_decay

    def update(self):
        self.sigma = max(self.sigma_min, self.sigma * self.sigma_decay)
        self.theta = min(self.theta_max, self.theta * self.theta_grow)

    def __repr__(self):
        representation =(
            f'OrnsteinUhlenbeckActionNoise(mu={self.mu}, sigma={round(self.sigma, 4)}/{round(self.sigma_min, 4)}, '+
            f'theta={round(self.theta, 4)}/{round(self.theta_max, 4)})')
        return representation

class GaussianNoise:
    '''
    Simple gaussian noise.
    '''
    def __init__(self, 
                 size,
                 mu,
                 sigma):
        self.size = size
        self.mu = mu

    def sample(self):
        return np.random.normal(loc=self.mu, scale=self.sigma, size=self.size)

    def update(self):
        return
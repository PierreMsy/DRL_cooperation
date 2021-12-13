import numpy as np

class Mock_env:

    def __init__(self,
                 nbr_agents,
                 state_size, 
                 action_size,
                 action_min,
                 action_max) -> None:
            
            self.nbr_agents = nbr_agents
            self.state_size = state_size
            self.action_size = action_size
            self.action_min = action_min
            self.action_max = action_max

    @property
    def vector_observations(self):
        return np.random.normal(0, 2, size=(self.nbr_agents, self.state_size))
    
    def step(self, _):

        next_observations = np.random.normal(0, 2, size=(self.nbr_agents, self.state_size))
        rewards = np.random.choice([0, .1, -.01], p=[.95, .02, .03], size=self.nbr_agents)
        dones= [np.random.choice([True, False], p=[.85, .15])] * 2

        return next_observations, rewards, dones 

class Mock_buffer_config:

    def __init__(self,
                 type,
                 size) -> None:
        self.type = type
        self.size = size
        self.device = 'cpu'
        self.epsilon = 0 # no probability floor to ease testing
        self.alpha = 1

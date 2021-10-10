import numpy as np
import torch
import random
from collections import deque, namedtuple
from abc import abstractmethod

class BufferCreator:

    def __init__(self):
        self.builders = {
            'uniform': lambda config: UniformReplayBuffer(config)
        }
    
    def create(self, config):
        return self.builders[config.buffer_type](config)

def stack_batch_by_agent(batch):
    '''
    batch [np.ndarray] :  batch of argument of some experiences.
    - if batch is either reward or done, just stack the values.
    - if batch is a full action / obs / next_obs stack the values by agent.
    '''
    if isinstance(batch[0], np.ndarray):
        return _group_by_agent(batch)
    else:
        return np.vstack(batch)

def _group_by_agent(batch):
    '''
    batch : either a batch by agent: np 3 dimentional array. 
    if needed split by agent : 
        [A1,B1],[A2,B2], ... ,[An,Bn] =>[A1,A2,...,An],[B1,B2,...,Bn] 
    '''
    return np.squeeze(np.split(batch, batch.shape[1], axis=1))

class ReplayBuffer:
    def __init__(self, config):
        
        self.Experience = namedtuple('Experience',
         ['obs_full', 'action_full', 'reward', 'next_obs_full', 'done'])
        self.memory = deque(maxlen=config.buffer_size)
        self.config = config

    @abstractmethod
    def add(self, observations, actions, reward, next_observations, done):
        raise NotImplementedError("Please Implement this method")

    @abstractmethod
    def sample(self):
        raise NotImplementedError("Please Implement this method")    

    def __len__(self):
        return len(self.memory)

    def _convert_to_torch(self, x):
        return torch.from_numpy(x).float().to(self.config.device)

class UniformReplayBuffer(ReplayBuffer):

    def __init__(self, config):
        super().__init__(config)

    def add(self, obs_full, action_full, reward, next_obs_full, done):
        '''
        Create an experience tuple from one interaction and add it to the memory
        '''
        experience = self.Experience(
            obs_full=obs_full,
            action_full=action_full,
            reward=reward,
            next_obs_full=next_obs_full,
            done=done)
        self.memory.append(experience)

    def sample(self, batch_size=None):
        '''
        Random sample as much experiences as requested by the batch_size 
        return for each element of the experience a batch of data.
        '''
        if batch_size is None:
            batch_size = self.config.batch_size

        experiences = random.sample(self.memory, batch_size)
        batches_experience_args = list(zip(*experiences))

        observations_batch, actions_batch, rewards, next_observations_batch, dones =\
            (self._convert_to_torch(stack_batch_by_agent(
                    np.array(batch_arg))) for batch_arg in batches_experience_args)

        return observations_batch, actions_batch, rewards, next_observations_batch, dones


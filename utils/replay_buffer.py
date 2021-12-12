import torch
import numpy as np
import random
from collections import deque, namedtuple
from abc import abstractmethod

from marl_coop.utils.sumTree import SumTree

class BufferCreator:
    '''
    Factory that build the buffer parametrized in the config among the possibilities:
        - uniform : All experiences have an equal probability to be sampled.
        - prioritized : Experiences have a sample probability proportional
                        to the error when last seen.
        - prioritized_sumTree : Same as last but with a storage structure optimized
                        for large batch sizes.
    '''
    def __init__(self):
        self.builders = {
            'uniform': lambda config: UniformReplayBuffer(config),
            'prioritized': lambda config: PrioritizedReplayBuffer(config),
            'prioritized_sumTree': lambda config: PrioritizedSumTreeBuffer(config)
        }
    
    def create(self, config):
        return self.builders[config.type](config)

def convert_to_agent_tensor(batch, device='cpu'):
    '''
    Convert list of np arrays : [batch * agent * size]
    to list of torch tensors : [agent * batch * size]
    '''
    convert_to_torch = lambda x, device : torch.tensor(x).float().to(device)
    return [convert_to_torch(np.array(agent_args), device) for agent_args in zip(*batch)]

class ReplayBuffer:
    '''
    Abstract class that lay the structure of all replay buffers.
    '''
    def __init__(self, config):
        
        self.Experience = namedtuple('Experience',
         ['obs_full', 'action_full', 'reward', 'next_obs_full', 'done'])
        self.memory = deque(maxlen=config.size)
        self.config = config

    @abstractmethod
    def add(self, observations, actions, reward, next_observations, done):
        raise NotImplementedError("Please Implement this method")

    @abstractmethod
    def sample(self):
        raise NotImplementedError("Please Implement this method")    

    def __len__(self):
        return len(self.memory)

class UniformReplayBuffer(ReplayBuffer):
    '''
    Simplest replay buffer where all experiences have an equal probability to be sampled.
    '''
    def __init__(self, config):
        super().__init__(config)
        self.is_PER = False

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

    def sample(self, sample_size=None):
        '''
        Sample randomly as much experiences as requested by the batch_size.
        Return for each element of the experience a batch of data.
        '''
        if sample_size is None:
            sample_size = self.config.batch_size
        
        experiences = random.sample(self.memory, sample_size)
        batches_experience_args = list(zip(*experiences))

        observations_batch, actions_batch, rewards, next_observations_batch, dones = (
            convert_to_agent_tensor(batch_arg, self.config.device) 
                for batch_arg in batches_experience_args)

        return observations_batch, actions_batch, rewards, next_observations_batch, dones

class PrioritizedReplayBuffer(ReplayBuffer):
    '''
    Buffer that will sample more often the experiences that generated a large time difference
    error indicating a misjudgment by the critic.
    '''
    def __init__(self, config):
        '''
        The priorities have to be included in the experiences to adapt the approximation
        of the gradient using a non uniform sampling of the experiences.
        In addition, the priorities are stored separately to ease the sampling. 
        '''
        super().__init__(config)
        self.is_PER = True
        self.Experience = namedtuple('Experience',
            ['obs_full', 'action_full', 'reward', 'next_obs_full', 'done', 'priority'])
        self.buffer_priority = deque(maxlen=config.size) 
        
    def add(self, obs_full, action_full, reward, next_obs_full, done, td_error):
        '''
        Compute the priority using the td_error and the hyperparameters.
        Create an experience tuple from one interaction and add it to the memory.
        '''
        priority = (abs(td_error) + self.config.epsilon) ** self.config.alpha

        experience = self.Experience(
            obs_full=obs_full,
            action_full=action_full,
            reward=reward,
            next_obs_full=next_obs_full,
            done=done,
            priority=priority)
        self.memory.append(experience)
        self.buffer_priority.append(priority)

    def sample(self, sample_size=None):
        '''
        Sample as much experience as requested from a distribution whose sample probability
        is given by the calculated prioriries.
        Return for each element of the experience a batch of data.
        '''
        if sample_size is None:
            sample_size = self.config.batch_size

        experiences = random.choices(self.memory,
                                     weights=self.buffer_priority, cum_weights=None,
                                     k=sample_size)
        batches_experience_args = list(zip(*experiences))

        observations_batch, actions_batch, rewards, next_observations_batch, dones, priorities = (
            convert_to_agent_tensor(batch_arg, self.config.device) 
                for batch_arg in batches_experience_args)

        return observations_batch, actions_batch, rewards, next_observations_batch, dones, priorities

    def update_experiences_priority(self, errors):
        '''
        Not implemented yet.
        '''
        return

class PrioritizedSumTreeBuffer(ReplayBuffer):
    '''
    Buffer that will sample more often the experiences that generated a large time difference
    error indicating a misjudgment by the critic.
    Use a structure called a sum tree to store the experiences that is optimized for sampling
    non-uniform distribution.
    '''
    def __init__(self, config):
        '''
        Indexes of the sampled experience have to be store to allow the update of the priority
        of those experiences.
        '''
        super().__init__(config)
        self.is_PER = True
        self.Experience = namedtuple('Experience',
            ['obs_full', 'action_full', 'reward', 'next_obs_full', 'done', 'priority'])
        self.memory = SumTree(tree_size=config.size)
        self.sample_with_replacement = config.sample_with_replacement
        self.sampled_idxs = ()

    def add(self, obs_full, action_full, reward, next_obs_full, done, error):
        '''
        Compute the priority using the td_error and the hyperparameters.
        Create an experience tuple from one interaction and add it to the memory.
        '''
        priority = self._compute_priority(error)

        experience = self.Experience(
            obs_full=obs_full,
            action_full=action_full,
            reward=reward,
            next_obs_full=next_obs_full,
            done=done,
            priority=priority)

        self.memory.add(experience, priority)

    def sample(self, sample_size=None):
        '''
        Sample as much experience as requested from a distribution whose sample probability
        is given by the calculated priorities.
        Return for each element of the experience a batch of data.
        '''
        if not sample_size:
            sample_size = self.config.batch_size
        
        idxs, experiences = self.memory.sample(sample_size, replacement=self.sample_with_replacement)
        self.sampled_idxs = idxs

        batches_experience_args = list(zip(*experiences))
        observations_batch, actions_batch, rewards, next_observations_batch, dones, priorities = (
            convert_to_agent_tensor(batch_arg, self.config.device) 
                for batch_arg in batches_experience_args)

        return observations_batch, actions_batch, rewards, next_observations_batch, dones, priorities

    def update_experiences_priority(self, errors):
        '''
        Update the priorities of the last sampled experience from the given td-errors.
        '''
        errors = errors.numpy()
        if len(self.sampled_idxs) != len(errors):
            raise Exception('Error updating the experiences priorities: '+
                           f'indexes len = {len(self.sampled_idxs)}, errors len = {len(errors)}')

        priorities = [self._compute_priority(error) for error in errors]
        self.memory.update_priorities(self.sampled_idxs, priorities)

    def _compute_priority(self, error):
        return (abs(error) + self.config.epsilon) ** self.config.alpha

    def __len__(self):
        return len([x for x in self.memory.values if isinstance(x, self.Experience)])
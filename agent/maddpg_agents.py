import numpy as np
import torch
from marl_coop import agent

from marl_coop.agent import DDPG_agent

# TODO PER and "custom tree search"
# TODO from tensorboardX import SummaryWriter
# TODO import imageio To save gif


class MADDPG_agents():
    """

    """

    def __init__(self, context, config) -> None:
        
        self.context = context
        self.config = config

        self._agents_named = {
            agent_name: DDPG_agent(context, agent_config, agent_idx, self) 
                for agent_idx, (agent_name, agent_config) in enumerate(config.agents.items())}
        self._agents = tuple(
            agent for _, agent in self.agents_named.items())

    @property
    def agents(self):  
        return self._agents

    @property
    def agents_named(self):  
        return self._agents_named

    def act(self, observations, noise=False):
        actions = np.array(
            [agent.act(obs, noise) for agent, obs in zip(self.agents, observations)])
        return actions

    def _act_target(self, obss_full):
        '''
        Used in agent learn to compute the TD target
        '''
        actions = [agent._act_target(obss) for agent, obss in zip(self.agents, obss_full)]
        return actions

    def step(self, obs_full, action_full, rewards, next_obs_full, dones):
        '''
        we store in the replay buffer experience as : 
        (x, a1, ... ,an, r, x', done) with x the observations of all the agents.
        ''' 
        for agent, reward, done in zip(self.agents, rewards, dones):
            agent.step(obs_full, action_full, np.array([reward]), next_obs_full, np.array([done]))

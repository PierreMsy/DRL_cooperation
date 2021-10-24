import numpy as np
import torch
from marl_coop import agent

from marl_coop.agent import DDPG_agent


class Selfplay_agent():
    """

    """

    def __init__(self, context, config) -> None:
        
        self.context = context
        self.config = config
        agent_cfg =  list(config.agents.values())[0]
        self._agent = DDPG_agent(context, agent_cfg, 0, self) 
        self._agents_named = {'agent' : self._agent}

    @property
    def agent(self):  
        return self._agent

    @property
    def agents(self):  
        return list(self._agent)

    @property
    def agents_named(self):  
        return self._agents_named

    def act(self, obss_full, noise=False):
        actions = np.array([self.agent.act(obs, noise) for  obs in obss_full])
        return actions

    def _act_target(self, obss_full):
        '''
        Used in agent learn to compute the TD target
        '''
        actions = [self.agent._act_target(obss) for obss in  obss_full]
        return actions

    def step(self, obs_full, action_full, rewards, next_obs_full, dones):
        '''
        we store in the replay buffer experience as : 
        (x, a1, ... ,an, r, x', done) with x the observations of all the agents.
        ''' 
        for reward, done in zip(rewards, dones):
            self.agent.step(obs_full, action_full, np.array([reward]), next_obs_full, np.array([done]))

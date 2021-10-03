import numpy as np
import torch

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

        # copy ou référence ?
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

    # TODO clean method
    def act(self, observations, noise=False):
        
        actions = []
        for agent, obs in zip(self.agents, observations):
            action = agent.act(obs, noise)
            actions.append(action)

        return np.array(actions)

    # TODO clean method
    def _act_target(self, observations):
        '''
        Used in agent learn to compute the TD target
        '''
        actions = torch.stack([
            agent._act_target(obs) for agent, obs in zip(self.agents, observations)])

        return actions

    def step(self, observations, actions, rewards, next_observations, dones):
        '''
        we store in the replay buffer experience as : 
        (x, a1, ... ,an, r, x', done) with x the observations of all the agents.
        '''

        for agent, reward, done in zip(self.agents, rewards, dones):
            agent.step(observations, actions, reward, next_observations, done)

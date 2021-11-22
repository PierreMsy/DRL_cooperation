import numpy as np

from marl_coop.agent import DDPG_agent


class MADDPG_agents():
    '''
    Multi-Agent Deep Deterministic Policy Gradient is a extension of DDPG in the MARL context.
    It use a decentralized execution and a centralized training where the critic is augmented with additional
    information about the policies of other agents.
    It leads to learned policies that only use local information at execution time, makes minimal assumption about
    the environment and the communicaiton between agents and is applicable to cooperative, competitive or mixed interaction.

    This class encapsulate all the ddpg agents and ease the interaction with them.
    '''
    def __init__(self, context, config) -> None:
        '''
        Instantiate all sub-agents with the configuration provided.
        '''
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
        Return a batch of actions for each agent batch of observations using target networks of the sub-agents.
        Used to compute the TD target.
        Args: 
            obss_full (list[torch tensor]): a batch of observations by agent.
        Returns:
             actions (list[torch tensor]): a batch of actions by agent.
        '''
        actions = [agent._act_target(obss) for agent, obss in zip(self.agents, obss_full)]
        return actions

    def step(self, obs_full, action_full, rewards, next_obs_full, dones):
        '''
        Store an interaction as a experience tuple and make the sub-agents learn when required.  
        Experience are stored in the replay buffer as: 
        (x, a1, ... ,an, r, x', done) with x the observations of all the agents.
        
        Args:
            obs_full (numpy.ndarray): the local observation of each sub-agent.
            action_full (numpy.ndarray): the action taken by each sub-agent.
            rewards (list[float]): the reward obtain by each sub-agent.
            next_obs_full (numpy.ndarray): the next local observation of each sub-agent.
            dones (list[bool]): a indicator of whether or not the next observation is a terminal
                state for each agent.   
        ''' 
        for agent, reward, done in zip(self.agents, rewards, dones):
            agent.step(obs_full, action_full, np.array([reward]), next_obs_full, np.array([done]))

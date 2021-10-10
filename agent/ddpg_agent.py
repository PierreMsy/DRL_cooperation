import numpy as np
from numpy.lib.function_base import select
import torch

from marl_coop.agent import Base_agent
from marl_coop.model import Actor_network_creator, Critic_network_creator
from marl_coop.utils import to_np, BufferCreator, CriterionCreator, NoiseCreator


class DDPG_agent(Base_agent):

    """
    Deep deterministic policy gradient agent.
    An agent will interact with the environnment to maximize the expected reward.
    This agent use a model-free, off-policy actor-critic algorithm using deep function approximators
    that can learn policies in high-dimensional, continuous action spaces.
    """

    def __init__(self, context, config, index, maddpg) -> None:
        
        self.context = context
        self.config = config
        self.index = index
        self.maddpg = maddpg

        self.buffer = BufferCreator().create(config)
        self.t_step = 0

        self.noise = NoiseCreator().create(config.noise.method,
            context.action_size, config.noise.kwargs)

        self.actor_network = Actor_network_creator().create(context, config.actor)
        self.actor_target_network = Actor_network_creator().create(context, config.actor)
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())

        self.critic_network = Critic_network_creator().create(context, config.critic)
        self.critic_target_network = Critic_network_creator().create(context, config.critic)
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())
        self.critic_criterion = CriterionCreator().create(config.critic.criterion)

    def act(self, observation, noise=False):

        observation = torch.from_numpy(observation).float().to(self.config.device)

        self.actor_network.eval()
        with torch.no_grad():
            action = self.actor_network.forward(observation)
        self.actor_network.train()
        action = to_np(action)
        
        if noise:
            action = np.clip(action + self.noise.sample(),
                self.context.action_min, self.context.action_max)

        return action

    def step(self, obs_full, action_full, reward, next_obs_full, done):
        '''
        Each centralized action-value function is learned independently using
        experiences defined as (x, a1, ... ,an, r, x', done).
        '''
        self.t_step +=1
        self.buffer.add(obs_full, action_full, reward, next_obs_full, done)

        if (len(self.buffer) >= self.config.batch_size) & (self.t_step % self.config.update_every == 0):
            self.learn()

    def learn(self):
        '''
        Sample experiences from the replay buffer and updates the critic and the actor.

        - The critic is updates based uppon a temporal difference error of the state-action value function
          using the actor to compute the actionsfrom the next state.
          error to minimize w.r.t w : r + γ * Q'_w'(s(t+1), µ'_θ'(s(t+1))) - Q_w(s(t),a(t))

        - The actor is updated using direct approximates of the state-action values from the critic.
          value to maximize w.r.t θ : Q_w(s(t), µ_θ(s(t+1)))  
        '''
        obss_full_batch, actions_full, rewards, next_obss_full, dones =\
            self.buffer.sample()

        next_actions_full = self.maddpg._act_target(obss_full_batch)

        # bug
        TD_targets = rewards + self.config.gamma * \
            self.critic_target_network(next_obss_full, next_actions_full) * (1 - dones)

        Q_values = self.critic_network(obss_full_batch, actions_full)

        loss = self.critic_criterion(Q_values, TD_targets)

        self.critic_network.optimizer.zero_grad()
        loss.backward()
        self.critic_network.optimizer.step()

        # see if it work
        action_pred = self.actor_network(obss_full_batch[self.index])
        next_actions_full[self.index] = action_pred
        Q_values = self.critic_target_network(next_obss_full, actions_full) 
        loss = - (Q_values).mean()

        self.actor_network.optimizer.zero_grad()
        loss.backward()
        self.actor_network.optimizer.step()

        soft_update(self.actor_target_network, self.actor_network, self.config.tau)
        soft_update(self.critic_target_network, self.critic_network, self.config.tau)

    def _act_target(self, observation, noise=False):
        '''
        Used in learn to compute the TD target
        observation : tensor
        '''
        self.actor_target_network.eval()
        action = self.actor_target_network.forward(observation)
        self.actor_target_network.train()

        return action
        
def soft_update(target_network, netwok, tau):
    '''
    net_weights = (1-τ) * net_weights + τ * target_net_weights 
    ''' 
    for target_param, local_param in zip(target_network.parameters(), netwok.parameters()):
        target_param.data.copy_(
            (1.0 - tau) * target_param.data + tau * local_param.data)


    
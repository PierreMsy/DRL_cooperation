import numpy as np
import torch

from marl_coop.agent import Base_agent
from marl_coop.model import Actor_network_creator, Critic_network_creator
from marl_coop.model.component import CriterionCreator
from marl_coop.component import BufferCreator, NoiseCreator
from marl_coop.utils.helper import to_np


class DDPG_agent(Base_agent):
    """
    Deep deterministic policy gradient agent.
    An agent will interact with the environnment to maximize the expected return.
    This agent use a model-free, off-policy actor-critic algorithm using deep function approximators
    that can learn policies in high-dimensional, continuous action spaces.
    """
    def __init__(self, context, config, index, maddpg):
        '''
        Store the config and the context.
        Instantiate the utilities: the noise function and the replay buffer.
        Instantiate the critic and the actor networks.

        Args:
            context : RL information such as state & action size.
            config : configuration of the agent and all its subparts.
            index : number associated with this agent.
            maddpg : a reference to the meta-agent.
        '''
        
        self.context = context
        self.config = config
        self.critic_loss = list()

        self.index = index
        self.maddpg = maddpg
        self.t_step = 0

        self.buffer = BufferCreator().create(config.buffer)
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
        '''
        Use the actor network to return a action from an observation with or
        without noise according to the noise parameter.
        '''
        observation = torch.from_numpy(observation).float().to(self.config.device)

        self.actor_network.eval()
        with torch.no_grad():
            action = self.actor_network.forward(observation)
        self.actor_network.train()
        action = to_np(action)
        
        if noise:
            action = np.clip(action + self.noise.sample(),
                self.context.action_min, self.context.action_max)
            self.noise.update()

        return action

    def step(self, obs_full, action_full, reward, next_obs_full, done):
        '''
        Each centralized action-value function is learned independently using
        experiences defined as (x, a1, ... ,an, r, x', done).
        If the buffer used is a prioritized experience replay, add the td error the 
        experience tuple to compute the priority.
        '''
        self.t_step +=1

        if self.buffer.is_PER:
            td_error  = self._compute_td_error(obs_full, action_full, reward, next_obs_full, done)
            self.buffer.add(obs_full, action_full, reward, next_obs_full, done, np.array([td_error]))

            if (len(self.buffer) >= self.config.buffer.batch_size) & (self.t_step % self.config.update_every == 0):
                for _ in range(self.config.learing_per_update):
                    self.learn_with_PER()
        else:
            self.buffer.add(obs_full, action_full, reward, next_obs_full, done)

            if (len(self.buffer) >= self.config.buffer.batch_size) & (self.t_step % self.config.update_every == 0):
                for _ in range(self.config.learing_per_update):
                    self.learn()
                
    def learn(self):
        '''
        Sample experiences from the replay buffer and updates the critic and the actor.
        
        The critic use a concatenation of the observations/actions from all agents to learn how to evaluate
        a situation in a stationary setting as describe in https://arxiv.org/pdf/1706.02275.pdf.
        The actor only operate on local data.

        - The critic is updates based uppon a temporal difference error of the state-action value function
          using the actor to compute the action from the next state.
          error to minimize w.r.t w : r + γ * Q'_w'(o(t+1), µ'_θ'(o(t+1))) - Q_w(o(t),a(t))

        - The actor is updated using direct approximates of the state-action values from the critic.
          value to maximize w.r.t θ : Q_w(o(t), µ_θ(o(t)))  
        '''
        obss_full, actions_full, rewards, next_obss_full, dones =\
            self.buffer.sample()

        # from a list of a flat array to an array of array of one element.
        rewards = rewards[0].view(-1,1)
        dones = dones[0].view(-1,1)

        next_actions_full = self.maddpg._act_target(next_obss_full)
        with torch.no_grad():
            Q_value_nexts = self.critic_target_network(next_obss_full, next_actions_full)
        Q_value_targets = rewards + self.config.gamma * Q_value_nexts * (1 - dones)

        Q_values = self.critic_network(obss_full, actions_full)

        critic_loss = self.critic_criterion(Q_values, Q_value_targets)
        self.critic_network.optimizer.zero_grad()
        critic_loss.backward()
        if self.config.use_gradient_clipping:
            torch.nn.utils.clip_grad_norm_(self.critic_network.parameters(), self.config.use_gradient_clipping)
        self.critic_network.optimizer.step()

        # actions have to be recomputed because the computation graph of the next actions has be thrown away
        actions_taken_full = [
            agent.actor_network(obss) if agent.index == idx  else  agent.actor_network(obss).detach()
                for idx, (agent, obss) in enumerate(zip(self.maddpg.agents, obss_full))]

        Q_values = self.critic_target_network(obss_full, actions_taken_full) 
        actor_loss = - (Q_values).mean()

        self.actor_network.optimizer.zero_grad()
        actor_loss.backward()
        self.actor_network.optimizer.step()

        soft_update(self.actor_target_network, self.actor_network, self.config.tau)
        soft_update(self.critic_target_network, self.critic_network, self.config.tau)

    def learn_with_PER(self):
        '''
        Sample experiences from the replay buffer and updates the critic and the actor.
        
        The critic use a concatenation of the observations/actions from all agents to learn how to evaluate
        a situation in a stationary setting as describe in https://arxiv.org/pdf/1706.02275.pdf.
        The actor only operate on local data.

        - The critic is updates based uppon a temporal difference error of the state-action value function
          using the actor to compute the action from the next state.
          error to minimize w.r.t w : r + γ * Q'_w'(o(t+1), µ'_θ'(o(t+1))) - Q_w(o(t),a(t))

        - The actor is updated using direct approximates of the state-action values from the critic.
          value to maximize w.r.t θ : Q_w(o(t), µ_θ(o(t)))

        Since the distrubution of the sampled experience won't match the real distribution with a PER,
        the stochastic gradient descent updates must be adjusted with the sampling weights: (1/N * 1/priority) ** β 
        '''
        obss_full, actions_full, rewards, next_obss_full, dones, priorities =\
            self.buffer.sample()

        # from a list of a flat array to an array of array of one element.
        rewards = rewards[0].view(-1,1)
        dones = dones[0].view(-1,1)
        priorities = priorities[0].view(-1,1)

        gradient_correction = (1/self.config.buffer.batch_size * 1/priorities).pow(self.config.buffer.beta)

        next_actions_full = self.maddpg._act_target(next_obss_full)
        with torch.no_grad():
            Q_value_nexts = self.critic_target_network(next_obss_full, next_actions_full)
        TD_targets = rewards + self.config.gamma * Q_value_nexts * (1 - dones)

        Q_values = self.critic_network(obss_full, actions_full)
        with torch.no_grad():
            errors = (Q_values * gradient_correction) - (TD_targets * gradient_correction)
        critic_loss = self.critic_criterion(Q_values * gradient_correction, TD_targets * gradient_correction)
        self.buffer.update_experiences_priority(errors)

        self.critic_network.optimizer.zero_grad()
        critic_loss.backward()
        if self.config.use_gradient_clipping:
            torch.nn.utils.clip_grad_norm_(self.critic_network.parameters(), self.config.use_gradient_clipping)
        self.critic_network.optimizer.step()

        # actions have to be recomputed because the computation graph of the next actions has be thrown away
        actions_taken_full = [
            agent.actor_network(obss) if agent.index == idx  else  agent.actor_network(obss).detach()
                for idx, (agent, obss) in enumerate(zip(self.maddpg.agents, obss_full))]
                
        Q_values = self.critic_target_network(obss_full, actions_taken_full) 
        actor_loss = - (gradient_correction * Q_values).mean()

        self.actor_network.optimizer.zero_grad()
        actor_loss.backward()
        self.actor_network.optimizer.step()

        soft_update(self.actor_target_network, self.actor_network, self.config.tau)
        soft_update(self.critic_target_network, self.critic_network, self.config.tau)

    def _act_target(self, observation):
        '''
        Return an action from an observation using the target network. Used to compute the TD target.
        Args: 
            observation (torch tensor)
        Returns:
             action (torch tensor)
        '''
        self.actor_target_network.eval()
        action = self.actor_target_network.forward(observation)
        self.actor_target_network.train()

        return action

    def _compute_td_error(self, obs_full, action_full, reward, next_obs_full, done):
        '''
        Compute the time difference associated with an experience tuple in order to obtain
        its priority of sampling.
        Returns:
            (float): the critic error for that experience.
        '''
        def _convert_to_torch(args):
            return [torch.from_numpy(np.expand_dims(arg, axis=0))
                        .float().to(self.config.device) for arg in args]
        obs_full = _convert_to_torch(obs_full)
        action_full = _convert_to_torch(action_full)
        next_obs_full = _convert_to_torch(next_obs_full)
        reward = reward[0]
        done = done[0]

        self.critic_network.eval()
        self.actor_network.eval()
        with torch.no_grad():
            Q_value = self.critic_target_network(obs_full, action_full)
            next_action_full = self.maddpg._act_target(next_obs_full)
            TD_target = reward + self.config.gamma * \
                self.critic_network(next_obs_full, next_action_full) * (1 - done)
        self.actor_network.train()
        self.critic_network.train()

        TD_error = (Q_value - TD_target).item()
        return TD_error
        
def soft_update(target_network, netwok, tau):
    '''
    net_weights = (1-τ) * net_weights + τ * target_net_weights 
    ''' 
    for target_param, local_param in zip(target_network.parameters(), netwok.parameters()):
        target_param.data.copy_(
            (1.0 - tau) * target_param.data + tau * local_param.data)

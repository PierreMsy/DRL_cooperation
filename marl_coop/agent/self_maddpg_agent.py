import numpy as np
import torch

from marl_coop.agent import Base_agent
from marl_coop.model import Actor_network_creator, Critic_network_creator
from marl_coop.model.utils import CriterionCreator
from marl_coop.utils import BufferCreator, NoiseCreator, to_np


class SelfMADDPG_agent(Base_agent):
    '''
    Multiple Agent Deep Deterministic Gradient implementation using a single actor & critic.

    Attempt at easing the training of MADDPG agents in symetrical environments where the same actor and critic
    can be used after a rearangment of the input tensors.
    '''
    def __init__(self, context, config):
        '''
        Store the config and the context.
        Instantiate the utilities: the noise function and the replay buffer.
        Instantiate the critic and the actor networks.

        Args:
            context : RL information such as state & action size.
            config : configuration of the agent and all its subparts.
        '''
        self.context = context
        self.config = config

        self.buffer = BufferCreator().create(config.buffer)
        self.noise = NoiseCreator().create(
            config.noise.method, context.action_size, config.noise.kwargs)

        self.actor_network = Actor_network_creator().create(context, config.actor)
        self.actor_target_network = Actor_network_creator().create(context, config.actor)
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())

        self.critic_network = Critic_network_creator().create(context, config.critic)
        self.critic_target_network = Critic_network_creator().create(context, config.critic)
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())
        self.critic_criterion = CriterionCreator().create(config.critic.criterion)

        self.t_step = 0

    @property
    def agents_named(self):  
        return {'agent': self}

    def act(self, obss_full, noise=False):
        '''
        Take an action for each observations inputted.

        Args:
            obss_full (np.ndarray): [(batch) * nbr_agents * obs_size]
                observations (or batch of) of all agents.
            noise (bool): Add or not noise to the outputted actions 
        '''
        obss_full = torch.from_numpy(obss_full).float().to(self.config.device)

        self.actor_network.eval()
        with torch.no_grad():
            actions = np.array([to_np(self.actor_network.forward(obss)) for obss in obss_full])
        self.actor_network.train()

        if noise:
            actions = np.clip(actions + self.noise.sample_multipe(self.context.nbr_agents),
                self.context.action_min, self.context.action_max)
            self.noise.update()

        return actions

    def step(self, obs_full, action_full, reward_full, next_obs_full, done_full):
        '''
        Store an interaction as a experience tuple and make the sub-agents learn when required.  
        Experience are stored in the replay buffer as: 
        (x, a1, ... ,an, r, x', done) with x the observations of all the agents.
        '''
        # transform [1,0,1] into [[1],[0],[1]]
        done_full =  np.array(done_full).reshape(-1,1)
        reward_full =  np.array(reward_full).reshape(-1,1)

        if self.config.buffer.type == 'prioritized':
            raise Exception('not yet implemented')
        else:
            self.buffer.add(obs_full, action_full, reward_full, next_obs_full, done_full)
            if (len(self.buffer) >= self.config.batch_size) & (self.t_step % self.config.update_every == 0):
                self.learn()

    def learn(self):
        '''
        Sample experiences from the replay buffer and updates the critic and the actor.
        
        The critic use a concatenation of the observations/actions from all agents to learn how to evaluate
        a situation in a stationary setting as describe in https://arxiv.org/pdf/1706.02275.pdf.
        The actor only operate on local data.

        For each agent a view of the data according to that agent is created resulting in 
        [nbr_agents * batch_size] updates at each learning.

        - The critic is updates based uppon a temporal difference error of the state-action value function
          using the actor to compute the action from the next state.
          error to minimize w.r.t w : r + γ * Q'_w'(o(t+1), µ'_θ'(o(t+1))) - Q_w(o(t),a(t))

        - The actor is updated using direct approximates of the state-action values from the critic.
          value to maximize w.r.t θ : Q_w(o(t), µ_θ(o(t)))  
        '''
        obss_full, actions_full, rewards, next_obss_full, dones = self.buffer.sample()
        dones = torch.stack(dones)
        rewards = torch.stack(rewards)

        next_actions_full = [self.actor_target_network.forward(obss) for obss in obss_full]
        next_obss_by_agent = _create_view_by_agent(next_obss_full, self.context.nbr_agents)
        next_actions_by_agent = _create_view_by_agent(next_actions_full, self.context.nbr_agents)
        with torch.no_grad():
            Q_value_nexts = self.critic_target_network(next_obss_by_agent, next_actions_by_agent)
        Q_value_targets = rewards + self.config.gamma * Q_value_nexts * (1 - dones)

        obss_by_agent = _create_view_by_agent(obss_full, self.context.nbr_agents)
        actions_by_agent = _create_view_by_agent(actions_full, self.context.nbr_agents)
        Q_values = self.critic_network(obss_by_agent, actions_by_agent)

        critic_loss = self.critic_criterion(Q_values, Q_value_targets)
        self.critic_network.optimizer.zero_grad()
        critic_loss.backward()
        if self.config.use_gradient_clipping:
            torch.nn.utils.clip_grad_norm_(self.critic_network.parameters(), 1)
        self.critic_network.optimizer.step()

        actions_taken_full = [self.actor_target_network.forward(obss) for obss in obss_full]
        actions_taken_by_agent = _create_view_by_agent(actions_taken_full, self.context.nbr_agents)
        Q_values = self.critic_target_network(obss_by_agent, actions_taken_by_agent) 

        actor_loss = - (Q_values).mean()
        self.actor_network.optimizer.zero_grad()
        actor_loss.backward()
        self.actor_network.optimizer.step()

        soft_update(self.actor_target_network, self.actor_network, self.config.tau)
        soft_update(self.critic_target_network, self.critic_network, self.config.tau)

def soft_update(target_network, netwok, tau):
    '''
    net_weights = (1-τ) * net_weights + τ * target_net_weights 
    ''' 
    for target_param, local_param in zip(target_network.parameters(), netwok.parameters()):
        target_param.data.copy_(
            (1.0 - tau) * target_param.data + tau * local_param.data)

def _create_view_by_agent(x, nbr_agents):
    '''
    Create a view of a list of tensors by agent by perfrming a cyrcular permutation of the data :
    [tensors(agent_0), tensors(agent_1), ..., tensors(agent_n)] =>
        for agent 0 :  tensors(tensors(agent_0) + tensors(agent_1) + ... + tensors(agent_n))
        for agent 1 :  tensors(tensors(agent_1) + tensors(agent_2) + ... + tensors(agent_0))
        ... 
        for agent n :  tensors(tensors(agent_n) + tensors(agent_0) + ... + tensors(agent_n-1))

    Args:
        x (list[torch.tensor]): list of tensors to arrange to create a view by agent.
        nbr_agents (int): number of agents.

    Returns:
        list[torch.tensor]: One arrangment of tensors where the data of the agent is first for each agents.
    '''
    res = [
        torch.roll(torch.stack(x), i, dims=0) 
            for i in range(nbr_agents)]
    return res

    
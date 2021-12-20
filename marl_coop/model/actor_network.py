import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from marl_coop.model.component import OptimizerCreator

#TODO : Modularize the size of the network
#TODO : Modularize the last activation function

class Actor_network_creator:

    def __init__(self):
        self.builders = {
            'vanilla': lambda context, config: Actor_network_vanilla(context, config),
            'batch_norm': lambda context, config: Actor_network_BN(context, config),
            }
    
    def create(self, context, config):
        return self.builders[config.architecture](context, config)

class Actor_network_BN(nn.Module):
    """
    function approximation that will directly learn the policy.
    Using the critic, this network will learn a deteministic mapping
    between the states and the actions.
    """
    
    def __init__(self, context, config):
        super(Actor_network_BN, self).__init__()
        
        self.config = config
        self.seed = torch.manual_seed(config.seed)

        self.fc1 = nn.Linear(context.state_size, config.hidden_layers[0])
        self.bn1 = nn.LayerNorm(config.hidden_layers[0])
        self.fc2 = nn.Linear(config.hidden_layers[0], config.hidden_layers[1])
        self.bn2 = nn.LayerNorm(config.hidden_layers[1])
        self.fc_to_actions = nn.Linear(config.hidden_layers[1], context.action_size)
        self.initialize_parameters()

        self.optimizer = OptimizerCreator().create(
            config.optimizer, self.parameters(), config.optim_kwargs)

    def initialize_parameters(self):
        '''
        Initialize the weight and biais of all layers with a unifom distribution 
        of spread 1 / sqrt(layer_size)
        '''
        spread = 1 / np.sqrt(self.fc1.weight.data.size()[0])
        torch.nn.init.uniform_(self.fc1.weight.data, -spread, spread)
        torch.nn.init.uniform_(self.fc1.bias.data, -spread, spread)

        spread = 1 / np.sqrt(self.fc2.weight.data.size()[0])
        torch.nn.init.uniform_(self.fc2.weight.data, -spread, spread)
        torch.nn.init.uniform_(self.fc2.bias.data, -spread, spread)

        spread = self.config.last_layer_init
        torch.nn.init.uniform_(self.fc_to_actions.weight.data, -spread, spread)
        torch.nn.init.uniform_(self.fc_to_actions.bias.data, -spread, spread)

    def forward(self, state):

        features = self.bn1(self.fc1(state))
        features = F.relu(features)

        features = self.bn2(self.fc2(features))
        features = F.relu(features)
        
        actions = torch.tanh(self.fc_to_actions(features))

        return actions 

class Actor_network_vanilla(nn.Module):
    """
    function approximation that will directly learn the policy.
    Using the critic, this network will learn a deteministic mapping
    between the states and the actions.
    """
    
    def __init__(self, context, config):
        super(Actor_network_vanilla, self).__init__()
        
        self.config = config
        self.seed = torch.manual_seed(config.seed)

        self.fc1 = nn.Linear(context.state_size, config.hidden_layers[0])
        self.fc2 = nn.Linear(config.hidden_layers[0], config.hidden_layers[1])
        self.fc_to_actions = nn.Linear(config.hidden_layers[1], context.action_size)
        self.initialize_parameters()

        self.optimizer = OptimizerCreator().create(
            config.optimizer, self.parameters(), config.optim_kwargs)

    def initialize_parameters(self):
        '''
        Initialize the weight and biais of all layers with a unifom distribution 
        of spread 1 / sqrt(layer_size)
        '''
        spread = 1 / np.sqrt(self.fc1.weight.data.size()[0])
        torch.nn.init.uniform_(self.fc1.weight.data, -spread, spread)
        torch.nn.init.uniform_(self.fc1.bias.data, -spread, spread)

        spread = 1 / np.sqrt(self.fc2.weight.data.size()[0])
        torch.nn.init.uniform_(self.fc2.weight.data, -spread, spread)
        torch.nn.init.uniform_(self.fc2.bias.data, -spread, spread)

        spread = self.config.last_layer_init
        torch.nn.init.uniform_(self.fc_to_actions.weight.data, -spread, spread)
        torch.nn.init.uniform_(self.fc_to_actions.bias.data, -spread, spread)

    def forward(self, state):

        features = F.relu(self.fc1(state))
        features = F.relu(self.fc2(features))
        actions = torch.tanh(self.fc_to_actions(features))

        return actions 
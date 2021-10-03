import json
import os
from copy import deepcopy


##TODO Make it possible to override specifc agent architecture.

PATH_JSON_MADDPG = os.path.join(os.path.dirname(__file__),
    r'./maddpg_config.json')
PATH_JSON_DDPG = os.path.join(os.path.dirname(__file__),
    r'./ddpg_config.json')

class MADDPG_configuration:

    def __init__(self, nbr_agents, agents_config={}):

        with open(PATH_JSON_MADDPG, 'r') as json_config:
            # default configuration
            self.base_dict = json.load(json_config)
            # configuration with overrides
            self.dict = deepcopy(self.base_dict)

        self.agents = {}

        # update the agent config with potential overrides.
        update_dict(self.dict['agents'], agents_config)
        if len(self.dict['agents']) > nbr_agents:
            raise Exception(
                'Too many agents configuration specified :'+
                f'given [{self.dict["agents"].keys}] for {nbr_agents} agents'
               )

        # instantiate a config by agent
        agents_dict = iter(self.dict['agents'].items())
        for idx_agent in range(1, nbr_agents+1):
            dict_agent = DDPG_configuration().dict
            # Take at default DDPG dict config  
            agent_name, dic_json = next(agents_dict, (None,None))
            if dic_json:
                update_dict(dict_agent, dic_json)
            if not agent_name:
                agent_name = f'agent_{idx_agent}'
            update_dict(self.dict, {'agents':{agent_name: dict_agent}})

            # Update the agents property
            self.agents[agent_name] = DDPG_configuration(**dict_agent)
            
class AgentConfiguration:

    def set_attr(self, attr, value):
        """
        if the value is different than None, update the config dictionary
        with it, otherwise, take the default dict's value.
        """
        if value:
            setattr(self, attr, value)
            self.dict[attr] = value
        else:
            setattr(self, attr, self.base_dict[attr])

    def __init__(self,
                 base_dict_path=None,
                 seed=None,
                 device=None,
                 gamma=None,
                 tau=None,
                 batch_size=None,
                 update_every=None):

        if base_dict_path:
            with open(base_dict_path, 'r') as json_config:
                self.base_dict = json.load(json_config)
        else:
            self.base_dict = {}
        self.dict = deepcopy(self.base_dict)

        self.set_attr('seed', seed)
        self.set_attr('device', device)
        self.set_attr('gamma', gamma) # discount factor
        self.set_attr('tau', tau) # target net soft update rate
        self.set_attr('batch_size', batch_size)
        self.set_attr('update_every', update_every)

    def __str__(self):
        representation = f""" *** BASE ***
 device : {self.device}
 gamma : {self.gamma}
 tau : {self.tau}
 batch_size : {self.batch_size}
 update every : {self.update_every}"""
        return representation

class DDPG_configuration(AgentConfiguration):

    def __init__(self,
                 seed=None,
                 device=None,
                 gamma=None,
                 tau=None,
                 batch_size=None,
                 update_every=None,
                 buffer_size=None,
                 buffer_type=None,
                 critic={},
                 actor={},
                 noise={}):
        super().__init__(seed=seed,
            base_dict_path=PATH_JSON_DDPG,
            device=device, gamma=gamma, tau=tau,
            batch_size=batch_size, update_every=update_every)

        self.set_attr('buffer_size', buffer_size)
        self.set_attr('buffer_type', buffer_type)

        update_dict(self.dict['critic'], critic)
        self.critic = Critic_configuration(self.dict['critic'])
        update_dict(self.dict['actor'], actor)
        self.actor = Actor_configuration(self.dict['actor'])
        update_dict(self.dict['noise'], noise)
        self.noise = Noise_configuration(self.dict['noise'])
    
    def __str__(self):
        representation = f"""{super().__str__()}
 *** BUFFER ***
buffer_size : {self.buffer_size}
buffer_type : {self.buffer_type}
 *** ACTOR ***
{self.actor.__str__()}
  *** CRITIC  ***
{self.critic.__str__()}
  *** NOISE  ***
{self.noise.__str__()}
 """
        return representation

class Critic_configuration:

    def __init__(self, dict_config):
        
        self.seed = dict_config['seed']
        self.hidden_layers = dict_config['hidden_layers']
        self.last_layer_init = dict_config['last_layer_init']
        self.architecture = dict_config['architecture']
        self.criterion = dict_config['criterion']
        self.optimizer = dict_config['optimizer']
        self.optim_kwargs = {
            'lr' : dict_config['learning_rate']
        }
    
    def __str__(self):
        return f"""learning rate : {self.optim_kwargs['lr']}        
architecture : {self.architecture}"""

class Actor_configuration:

    def __init__(self, dict_config):
        
        self.seed = dict_config['seed']
        self.hidden_layers = dict_config['hidden_layers']
        self.last_layer_init = dict_config['last_layer_init']
        self.architecture = dict_config['architecture']
        self.optimizer = dict_config['optimizer']
        self.optim_kwargs = {
            'lr' : dict_config['learning_rate']
        }
    
    def __str__(self):
        return f"""learning rate : {self.optim_kwargs['lr']}
architecture : {self.architecture}"""

class Noise_configuration:

    def __init__(self, dict_config):

        self.method = dict_config['method']
        self.mu = dict_config['mu']
        self.sigma = dict_config['sigma']
        self.theta = dict_config['theta']

        self.kwargs = {
            'mu' : self.mu,
            'sigma' : self.sigma,
            'theta' : self.theta
        }

    def __str__(self):
        return f"""method : {self.method}"""

def update_dict(d_ref, d_ovr):
    """
    for every specific kv given in d_ovr, change the corresponding
    values in d_ref, the complete and udpated config dictionary.
    """
    for k_o, v_o in d_ovr.items():
        if k_o not in d_ref:
            d_ref[k_o] = v_o

    for k_r,v_r in d_ref.items():
        if k_r in d_ovr:
            if type(v_r) == dict:
                update_dict(d_ref[k_r], d_ovr[k_r])
            else:
                d_ref[k_r] = d_ovr[k_r]
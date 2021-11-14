import json
import os
from copy import deepcopy


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
    
    @classmethod
    # Alternative "contructor" to load from output
    def create_from_dict(cls, dict):
        nbr_agents = len(dict['agents'])
        obj = cls.__new__(cls)
        cls.__init__(obj, nbr_agents, agents_config=dict['agents'])
        return obj
            
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
                 update_every=None,
                 use_gradient_clipping=None,
                 learing_per_update=None):

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
        self.set_attr('use_gradient_clipping', use_gradient_clipping)
        self.set_attr('learing_per_update', learing_per_update)
        

    def __str__(self):
        representation = f""" *** BASE ***
 device : {self.device}u
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
                 use_gradient_clipping=None,
                 learing_per_update=None,
                 buffer={},
                 critic={},
                 actor={},
                 noise={}):
        super().__init__(seed=seed,
            base_dict_path=PATH_JSON_DDPG, device=device,
            gamma=gamma, tau=tau, batch_size=batch_size, 
            update_every=update_every,
            use_gradient_clipping=use_gradient_clipping,
            learing_per_update=learing_per_update)

        self.dict['buffer']['batch_size'] = self.batch_size
        self.dict['buffer']['device'] = self.device
        update_dict(self.dict['buffer'], buffer)
        self.buffer = Buffer_configuration(self.dict['buffer'])

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
        if 'momentum' in dict_config:
            self.optim_kwargs['momentum'] = dict_config['momentum']
    
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
        if 'momentum' in dict_config:
            self.optim_kwargs['momentum'] = dict_config['momentum']
    
    def __str__(self):
        return f"""learning rate : {self.optim_kwargs['lr']}
architecture : {self.architecture}"""

class Buffer_configuration:

    def __init__(self, dict_config):
        self.size = dict_config['size']
        self.batch_size = dict_config['batch_size']
        self.device = dict_config['device']
        self.type = dict_config['type']
        self.alpha = dict_config.get('alpha')
        self.beta = dict_config.get('beta')
        self.epsilon = dict_config.get('epsilon')

        self.kwargs = {
            'size' : self.size,
            'batch_size' : self.batch_size,
            'device' : self.device,
            'type' : self.type,
        }
        if self.alpha:
            self.kwargs['alpha'] = self.alpha
            self.kwargs['beta'] = self.beta
            self.kwargs['epsilon'] = self.epsilon
    
    def __str__(self):
        return f"""type : {self.type}, size : {self.size}"""

class Noise_configuration:

    def __init__(self, dict_config):
        
        def convert_nbr_updates_to_update_scale(start, end, nbr):
            scale = (end/start)**(1/nbr)
            return scale

        self.method = dict_config.get('method')
        self.mu = dict_config.get('mu')
        self.theta = dict_config.get('theta')
        self.theta_max = dict_config.get('theta_max')
        self.theta_nbr_growths = dict_config.get('theta_nbr_growths')
        self.sigma = dict_config.get('sigma')
        self.sigma_min = dict_config.get('sigma_min')
        self.sigma_nbr_decays = dict_config.get('sigma_nbr_decays')

        self.kwargs = {
            'mu' : self.mu,
            'theta' : self.theta,
            'sigma' : self.sigma,
        }
        if self.theta_max:
            self.kwargs['theta_max'] = self.theta_max
            self.kwargs['theta_grow'] = convert_nbr_updates_to_update_scale(
                self.theta, self.theta_max, self.theta_nbr_growths)
                
        if self.sigma_min:
            self.kwargs['sigma_min'] = self.sigma_min
            self.kwargs['sigma_decay'] = convert_nbr_updates_to_update_scale(
                self.sigma, self.sigma_min, self.sigma_nbr_decays)

    def __str__(self):
        return f"""method : {self.method}"""

def update_dict(d_ref, d_ovr):
    """
    for every specific kv given in d_ovr, change the corresponding
    values in d_ref, the complete and udpated config dictionary.
    """
    try:
        for k_o, v_o in d_ovr.items():
            if k_o not in d_ref:
                d_ref[k_o] = v_o
    
        for k_r,v_r in d_ref.items():
            if k_r in d_ovr:
                if type(v_r) == dict:
                    update_dict(d_ref[k_r], d_ovr[k_r])
                else:
                    d_ref[k_r] = d_ovr[k_r]
    except:
        print('dommage...')
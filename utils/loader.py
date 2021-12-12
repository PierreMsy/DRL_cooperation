import os
import torch
import json

from marl_coop.utils import get_files
from marl_coop.config import MADDPG_configuration
from marl_coop.agent import MADDPG_agents

def load_agent(key, agent, path=None, day=None, month=None, year=None):

    if path is None:
        path = os.path.join(os.path.dirname(__file__), r'./../../output/model')
    files = get_files(path, key, day, month, year)

    if len(files) > 2:
        raise Exception('Please add temporal information, too much models matched the given key:\n'+
        f'Found files :{files}\nInput key : {key}, day : {day}, month : {month}, year : {year}.')

    actor_model = [f for f in files if 'actor' in f][0]
    actor_checkpoint = torch.load(os.path.join(path,actor_model))

    agent.actor_network.load_state_dict(actor_checkpoint)
    agent.actor_target_network.load_state_dict(actor_checkpoint)

    critic_model = [f for f in files if 'critic' in f][0]
    critic_checkpoint = torch.load(os.path.join(path, critic_model))

    agent.critic_network.load_state_dict(critic_checkpoint)
    agent.critic_target_network.load_state_dict(critic_checkpoint)

    return agent

def load_MADDPG_agent(key, context, path_cfg=None, path_model=None, day=None, month=None, year=None):

    if path_cfg is None:
        path_cfg = os.path.join(os.path.dirname(__file__), r'./../../output/configuration')
    if path_model is None:
        path_model = os.path.join(os.path.dirname(__file__), r'./../../output/model')
    
    file_cfg = get_files(path_cfg, key, day, month, year)
    if len(file_cfg) > 1:
        raise Exception('Please add temporal information, too much models matched the given key:\n'+
        f'Found files :{file_cfg}\nInput key : {key}, day : {day}, month : {month}, year : {year}.')
    file_cfg = file_cfg[0]

    with open(os.path.join(path_cfg, file_cfg), 'r') as json_config:
            dict_config = json.load(json_config)
    cfg = MADDPG_configuration.create_from_dict(dict_config)
    maddpg = MADDPG_agents(context, cfg)

    repository_models = get_files(path_model, key, day, month, year)
    if len(repository_models) > 1:
        raise Exception('Please add temporal information, too much models matched the given key:\n'+
        f'Found files :{repository_models}\nInput key : {key}, day : {day}, month : {month}, year : {year}.')
    repository_models = os.path.join(path_model, repository_models[0])
    
    files_model = os.listdir(os.path.join(path_model, repository_models))

    for agent_name, agent in maddpg.agents_named.items():

        actor_model = [f for f in files_model if ('actor' in f) & (agent_name in f)][0]
        actor_checkpoint = torch.load(os.path.abspath(os.path.join(repository_models, actor_model)))
    
        agent.actor_network.load_state_dict(actor_checkpoint)
        agent.actor_target_network.load_state_dict(actor_checkpoint)
    
        critic_model = [f for f in files_model if 'critic' in f][0]
        critic_checkpoint = torch.load(os.path.abspath(os.path.join(repository_models, critic_model)))
    
        agent.critic_network.load_state_dict(critic_checkpoint)
        agent.critic_target_network.load_state_dict(critic_checkpoint)

    return maddpg
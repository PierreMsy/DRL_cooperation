from marl_coop.config import MADDPG_configuration
import numpy as np

def test_MADDPG_config_can_be_instantiated_with_default_agents_config():
    
    config = MADDPG_configuration(nbr_agents=2)

    cfg_agents = config.dict['agents']
    
    assert 'agent_1' in cfg_agents
    assert 'agent_2' in cfg_agents
    assert type(cfg_agents['agent_1']) == dict
    assert type(cfg_agents['agent_2']) == dict
    assert len(config.agents) == 2


def test_MADDPG_config_can_be_instantiated_with_specific_config():
    
    override = {
            'agent_a':{},
            'agent_b':{}
    }

    config = MADDPG_configuration(nbr_agents=2, agents_config=override)

    cfg_agents = config.dict['agents']
    
    assert 'agent_a' in cfg_agents
    assert 'agent_b' in cfg_agents
    assert type(cfg_agents['agent_a']) == dict
    assert type(cfg_agents['agent_b']) == dict
    assert len(config.agents) == 2

def test_MADDPG_config_can_be_instantiated_with_specfic_DDPG_confg():

    agent_cfg = {
        'seed':1,
        'update_every':5,
        'gamma':.97,
        'tau':5e-4,
        'critic':{'learning_rate': 2e-4}
    }
    
    agents_config={
        'agent_A': agent_cfg,
        'agent_B': agent_cfg
    }

    cfg = MADDPG_configuration(nbr_agents=2, agents_config=agents_config)

    cfg_agents = cfg.dict['agents']
    assert 'agent_A' in cfg_agents
    assert 'agent_B' in cfg_agents
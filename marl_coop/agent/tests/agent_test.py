import pytest

from marl_coop.agent import MADDPG_agents
from marl_coop.config import MADDPG_configuration
from marl_coop.utils.context import Context
from marl_coop.test.mock import Mock_env


@pytest.fixture()
def nbr_agents():
    yield 2

@pytest.fixture()
def env(nbr_agents):
    env = Mock_env(nbr_agents=nbr_agents,
                   state_size=8,
                   action_size=2,
                   action_min=-1, action_max=1)
    return env

def test_that_maddpg_agents_can_act(nbr_agents, env):
    
    cfg = MADDPG_configuration(nbr_agents=nbr_agents)
    ctx = Context(env, {'nbr_agents': nbr_agents})

    maddpg = MADDPG_agents(ctx, cfg)

    observations = env.vector_observations
    actions = maddpg.act(observations)

    assert len(actions) == 2

def test_that_maddpg_agents_can_step(nbr_agents, env):
    
    cfg = MADDPG_configuration(nbr_agents=nbr_agents)
    ctx = Context(env, {'nbr_agents': nbr_agents})

    maddpg = MADDPG_agents(ctx, cfg)

    observations = env.vector_observations
    
    for _ in range(5):

        actions = maddpg.act(observations)
        next_observations, rewards, dones = env.step(actions)
        maddpg.step(observations, actions, rewards, next_observations, dones)
        observations = next_observations

    assert len(actions) == 2


def test_that_maddpg_agents_can_learn(nbr_agents, env):
    
    agents_config = {
        'agent_1': {'update_every': 3, 'buffer':{'batch_size': 3}},
        'agent_2': {'update_every': 3, 'buffer':{'batch_size': 3}}
        }

    cfg = MADDPG_configuration(nbr_agents=nbr_agents, agents_config=agents_config)
    ctx = Context(env, {'nbr_agents': nbr_agents})

    maddpg = MADDPG_agents(ctx, cfg)

    observations = env.vector_observations
    
    # step
    for _ in range(2):
        actions = maddpg.act(observations)
        next_observations, rewards, dones = env.step(actions)
        maddpg.step(observations, actions, rewards, next_observations, dones)
        observations = next_observations

    # learn
    for _ in range(4):
        actions = maddpg.act(observations)
        next_observations, rewards, dones = env.step(actions)
        maddpg.step(observations, actions, rewards, next_observations, dones)
        observations = next_observations

    assert len(actions) == 2
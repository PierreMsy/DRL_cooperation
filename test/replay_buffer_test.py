import torch
import numpy as np

from marl_coop.utils.replay_buffer import stack_batch_by_agent
from marl_coop.utils import BufferCreator
from marl_coop.test.mock import Mock_config


def test_group_by_agent_convert_interleaved_agent_experience_into_batch_by_agent():
    
    arg = np.array([[
            ['A11','A12','A13'],
            ['B11','B12','B13'],
            ['C11','C12','C13']
        ],[
            ['A21','A22','A23'],
            ['B21','B22','B23'],
            ['C21','C22','C23']
        ],[
            ['A31','A32','A33'],
            ['B31','B32','B33'],
            ['C31','C32','C33']]])

    expected = np.array([[
            ['A11','A12','A13'],
            ['A21','A22','A23'],
            ['A31','A32','A33']
        ],[
            ['B11','B12','B13'],
            ['B21','B22','B23'],
            ['B31','B32','B33']
        ],[
            ['C11','C12','C13'],
            ['C21','C22','C23'],
            ['C31','C32','C33']]])

    actual = stack_batch_by_agent(arg)

    assert np.all(actual == expected) == True

def test_group_by_agent_convert_interleaved_agent_experience_into_batch_by_agent_even_not_square():
    
    arg = np.array([[
            ['A11','A12'],
            ['B11','B12']
        ],[
            ['A21','A22'],
            ['B21','B22']
        ],[
            ['A31','A32'],
            ['B31','B32']
        ],[
            ['A41','A42'],
            ['B41','B42']
            ]])

    expected = np.array([[
            ['A11','A12'],
            ['A21','A22'],
            ['A31','A32'],
            ['A41','A42'],
        ],[
            ['B11','B12'],
            ['B21','B22'],
            ['B31','B32'],
            ['B41','B42'],]])

    actual = stack_batch_by_agent(arg)

    assert np.all(actual == expected) == True

def test_buffer_sampling():

    buffer = BufferCreator().create(Mock_config(buffer_size=3, buffer_type='uniform'))

    observations_s = np.array([[[111,112,113],
                                [121,122,123]
                                ],[
                                [211,212,213],
                                [221,222,223]
                                ],[
                                [311,312,313],
                                [321,322,323]]])
    actions_s = observations_s.copy()
    reward_s = np.array([[0,1],[0,1],[0,1]])
    next_observations_s = observations_s.copy()
    done_s = np.array([[False,False],[False,False],[True,True]])

    for observation, action, reward, next_observation, done in zip(
        observations_s, actions_s, reward_s, next_observations_s, done_s):
        buffer.add(observation, action, reward, next_observation, done)

    observations_batch, _, _, _, _ = buffer.sample(3)
    observations_batch, _ = torch.sort(observations_batch, dim=1)

    expected_observations = torch.from_numpy(np.array([
        [[111,112,113],[211,212,213],[311,312,313]],
        [[121,122,123],[221,222,223],[321,322,323]]
    ])).float()

    assert torch.all(observations_batch == expected_observations) == True

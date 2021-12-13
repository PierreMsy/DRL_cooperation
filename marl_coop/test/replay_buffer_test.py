import torch
import numpy as np

from marl_coop.utils.replay_buffer import convert_to_agent_tensor
from marl_coop.utils import BufferCreator
from marl_coop.test.mock import Mock_buffer_config


def test_group_by_agent_convert_interleaved_agent_experience_into_batch_by_agent():
    
    arg = np.array([[
            [111, 112, 113],
            [211, 212, 213],
            [311, 312, 313]
        ],[
            [121, 122, 123],
            [221, 222, 223],
            [321, 322, 323]
        ],[
            [131, 132, 133],
            [231, 232, 233],
            [331, 332, 333]]])

    expected = [
        torch.tensor([
            [111., 112., 113.],
            [121., 122., 123.],
            [131., 132., 133.]
        ]),
        torch.tensor([
            [211., 212., 213.],
            [221., 222., 223.],
            [231., 232., 233.]
        ]),
        torch.tensor([
            [311., 312., 313.],
            [321., 322., 323.],
            [331., 332., 333.]])]

    actual = convert_to_agent_tensor(arg)

    assert np.all([torch.all(exp == act).numpy() for exp, act in zip(expected, actual)]) == True

def test_group_by_agent_convert_interleaved_agent_experience_into_batch_by_agent_even_not_square():
    
    arg = np.array([[
            [111, 112],
            [211, 212]
        ],[
            [121, 122],
            [221, 222]
        ],[
            [131, 132],
            [231, 232]
        ],[
            [141, 142],
            [241, 242]
            ]])

    expected = [
        torch.tensor([
            [111., 112.],
            [121., 122.],
            [131., 132.],
            [141., 142.],
        ]),
        torch.tensor([
            [211., 212.],
            [221., 222.],
            [231., 232.],
            [241., 242.]])]

    actual = convert_to_agent_tensor(arg)

    assert np.all([torch.all(exp == act).numpy() for exp, act in zip(expected, actual)]) == True

def test_buffer_sampling():

    buffer = BufferCreator().create(Mock_buffer_config(size=3, type='uniform'))

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
    observations_batch, _ = torch.sort(torch.stack(observations_batch), dim=1)

    expected_observations = torch.from_numpy(np.array([
        [[111,112,113],[211,212,213],[311,312,313]],
        [[121,122,123],[221,222,223],[321,322,323]]
    ])).float()

    assert torch.all(observations_batch == expected_observations) == True

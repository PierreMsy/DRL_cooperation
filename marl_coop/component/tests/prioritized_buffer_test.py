import numpy as np
import torch

from marl_coop.test.mock import Mock_buffer_config
from marl_coop.component import BufferCreator


def test_that_prioritized_sumTree_sample_experiences_wrt_erros():

    buffer = BufferCreator().create(Mock_buffer_config(size=3, type='prioritized_sumTree'))

    observations_s = [np.random.random((2,3)), np.random.random((2,3)), np.random.random((2,3))]
    actions_s = [np.array([0,0]), np.array([1,1]), np.array([2,2])]
    reward_s = np.array([[0,1],[0,1],[0,1]])
    next_observations_s = [np.random.random((2,3)), np.random.random((2,3)), np.random.random((2,3))]
    done_s = np.array([[False,False],[False,False],[True,True]])
    error_s = np.array([[0],[1],[0]])

    for observation, action, reward, next_observation, done, error in zip(
        observations_s, actions_s, reward_s, next_observations_s, done_s, error_s):
        buffer.add(observation, action, reward, next_observation, done, error)
    _, action_batch, _, _, _, _ = buffer.sample(3)

    expected_actions = [torch.from_numpy(np.array([1, 1, 1])).float() for i in range(2)]
    list_of_checks = [torch.all(action == expected_action) for action, expected_action in zip(action_batch, expected_actions)]

    assert np.all(np.array(list_of_checks)) == True


def test_that_prioritized_sumTree_experiences_can_be_updated():

    buffer = BufferCreator().create(Mock_buffer_config(size=3, type='prioritized_sumTree'))

    observations_s = [np.random.random((2,3)), np.random.random((2,3)), np.random.random((2,3))]
    actions_s = [np.array([0,0]), np.array([1,1]), np.array([2,2])]
    reward_s = np.array([[0,1],[0,1],[0,1]])
    next_observations_s = [np.random.random((2,3)), np.random.random((2,3)), np.random.random((2,3))]
    done_s = np.array([[False,False],[False,False],[True,True]])
    error_s = np.array([[0],[1],[0]])


    for observation, action, reward, next_observation, done, error in zip(
        observations_s, actions_s, reward_s, next_observations_s, done_s, error_s):
        buffer.add(observation, action, reward, next_observation, done, error)

    _, _, _, _, _, _ = buffer.sample(3) #sample to unable the priority update
    buffer.update_experiences_priority(torch.tensor([0,0,0])) # the second experience receive 0 as priority
    buffer.memory.update_priorities([0,2],[0,1]) # update exps 0 & 2 with priorities 0 & 1 
    _, action_batch, _, _, _, _ = buffer.sample(3)

    expected_actions = [torch.from_numpy(np.array([2, 2, 2])).float() for i in range(2)]
    list_of_checks = [torch.all(action == expected_action) for action, expected_action in zip(action_batch, expected_actions)]

    assert np.all(np.array(list_of_checks)) == True
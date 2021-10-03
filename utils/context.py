from unityagents import UnityEnvironment
from unityagents.brain import BrainParameters
import gym


from marl_coop.test.mock import Mock_env

class Context:

    def __init__(self, env, kwargs={}) -> None:

        if isinstance(env, UnityEnvironment):

            raise Exception('UnityEnvironment not supported yet')

        elif isinstance(env, BrainParameters):

            self.state_size = (
                env.vector_observation_space_size * env.num_stacked_vector_observations)
            self.action_size = env.vector_action_space_size
            self.action_min = kwargs.get("action_min")
            self.action_max = kwargs.get("action_max")

        elif isinstance(env, gym.wrappers.time_limit.TimeLimit):

            self.state_size = env.observation_space.shape
            if len(self.state_size) == 1:
                self.state_size = self.state_size[0]
            self.action_size = env.action_space.shape
            if len(self.action_size) == 1:
                self.action_size = self.action_size[0]
            self.action_min = env.action_space.low
            self.action_max = env.action_space.high

        elif isinstance(env, Mock_env):
            
            self.state_size = env.state_size
            self.action_size = env.action_size
            self.action_min = env.action_min
            self.action_max = env.action_max

        else:
            raise Exception('Environment provided must be either open gym'+
            'or Unity environnement')

        self.nbr_agents = kwargs.get("nbr_agents")

    def __repr__(self):
        representation = f""" *** CONTEXT ***
state_size : {self.state_size}
action_size : {self.action_size}
action_range : [{self.action_min}, {self.action_max}] 
 """
        return representation
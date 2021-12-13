import os

import numpy as np
from collections import deque

from marl_coop.utils import save_scores, save_AC_models, save_configuration


class Runner:
    '''
    Helper class that handle the training of agents in a MARL environment.
    '''
    def __init__(self) -> None:
        file_location = os.path.abspath(os.getcwd()) # os.path.dirname(__file__)
        self.path_score = os.path.join(file_location, r'./output/score')
        self.path_model = os.path.join(file_location, r'.//output/model')
        self.path_config = os.path.join(file_location, r'./output/configuration')
        
    def run(self, MA_agents, env, brain_name, nb_episodes, run_key,
            average_on=10, average_break=None, target_score=None, target_over=100,
            save_score=False, save_config=False, save_weights=False, save_interaction=False):
        '''
        Train multiple agents by making them interact with the provided environment.

        Args:
            MA_agents (MADDPG_agents): Multiple-agents RL algorithm
            env (UnityEnvironment/OpenAI gym): MARL environment
            brain_name ([type]): [description]
            nb_episodes (int): number of training episodes
            run_key (string): name of the training for recordings
            average_on (int, defaults to 10) : number of episode score to average to diplay performances
            average_break (int, optional) : average score that to stop the training onve reached
            target_score (int, optional): mean score to reach over a certain number of episodes
            target_over (int, defaults to 100): number of score to average to compare to the target score
            save_score (bool, defaults to False): option to save the score by agent by episode
            save_config (bool, defaults to False): option to save the configuration of the agents
            save_weights (bool, defaults to False): option to save the networks (critic & actor) by agent
            save_interaction (bool, defaults to False): option to save a gif of interaction of the agents
        '''
        self.log_sigma = 0
        scores = deque()
        score_averaged = float('-inf')
        scores_by_agent = {k:deque() for k in MA_agents.agents_named}

        scores_target = deque(maxlen=target_over)
        scores_window = deque(maxlen=average_on)
        is_solved = ''
        
        for episode in range(1, nb_episodes+1):
            
            if average_break and score_averaged >= average_break: 
                break
            env_info = env.reset(train_mode=True)[brain_name] 
            observations = env_info.vector_observations          
            score = 0
            score_by_agent=np.zeros(MA_agents.context.nbr_agents)
            
            while True:
                
                actions = MA_agents.act(observations, noise=True)
                
                env_info = env.step(actions)[brain_name]
                next_observations = env_info.vector_observations
                rewards = env_info.rewards
                dones = env_info.local_done
                
                MA_agents.step(observations, actions, rewards, next_observations, dones)
                
                observations = next_observations
                
                score += np.mean(rewards)
                score_by_agent += rewards
                
                if np.any(dones):
                    
                    self._print_log_noise_scaling(MA_agents, episode)
                    scores.append(score)
                    if hasattr(MA_agents, 'agents_named'):
                        for key,reward in zip(scores_by_agent.keys(), score_by_agent):
                            scores_by_agent[key].append(reward)

                    scores_target.append(score)
                    scores_window.append(score)
                    score_averaged = np.mean(list(scores_window))
                    print(f"\rEpisode {episode} Score: {round(score_averaged,5)}{is_solved}",
                        end='\r')
                    if target_score:
                        if (len(is_solved) == 0) & (np.mean(list(scores_target)) > target_score):
                            is_solved = f' -> Solved in {episode} episodes'
                    break
                    
        print(f"\nLast score: {round(score_averaged,5)} {is_solved}")  
                
        if save_score:
            save_scores(scores, scores_by_agent, run_key, self.path_score)
    
        if save_config:
            save_configuration(MA_agents, run_key, self.path_config)
    
        if save_weights:
            save_AC_models(MA_agents, run_key, self.path_model)
    
        if save_interaction:
            # not available for the version of unity environment used.
            raise Exception('not implemented yet')

    def _print_log_noise_scaling(self, MA_agents, episode):

        if not hasattr(MA_agents, 'agents'):
            return
        if not hasattr(MA_agents.agents[0].noise, 'sigma_init'):
            return

        sigma_init = MA_agents.agents[0].noise.sigma_init
        sigma_min = MA_agents.agents[0].noise.sigma_min 
        sigma = MA_agents.agents[0].noise.sigma
        sigma_half_life = ((sigma_init + sigma_min) / 2)

        if (self.log_sigma < .5)  & (sigma < sigma_half_life):
            self.log_sigma = .5
            print(f'Noise scaling half life reached in {episode} episodes : {MA_agents.agents[0].noise}')

        if (self.log_sigma < 1)  & (sigma <= sigma_min):
            self.log_sigma = 1
            print(f'Noise scaling complete in {episode} episodes : {MA_agents.agents[0].noise}')
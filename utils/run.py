import os

import numpy as np
from collections import deque

from marl_coop.utils import save_scores, save_AC_models, save_configuration


class Runner:

    def __init__(self) -> None:

        file_location = os.path.dirname(__file__)
        self.path_score = os.path.join(file_location, r'./../../output/score')
        self.path_model = os.path.join(file_location, r'./../../output/model')
        self.path_config = os.path.join(file_location, r'./../../output/configuration')

    def run(self, MA_agents, env, brain_name, nb_episodes, run_key,
            average_on=10, target_score=None, target_over=100,
            save_score=True, save_config=True, save_weights=False, save_interaction=False):
            
            scores = deque()
            scores_by_agent = {k:deque() for k in MA_agents.agents_named}

            scores_target = deque(maxlen=target_over)
            scores_window = deque(maxlen=average_on)
            is_solved = ''
            
            for episode in range(1, nb_episodes+1):
                
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

                        if (episode % 1000 == 0) & hasattr(MA_agents, 'agents') & hasattr(MA_agents.agents[0].noise, 'sigma_init'):
                            if MA_agents.agents[0].noise.sigma != MA_agents.agents[0].noise.sigma_init:
                                print(MA_agents.agents[0].noise)

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
                raise Exception('not implemented yet')
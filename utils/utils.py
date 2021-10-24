import os
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from datetime import datetime
import json

import torch
import torch.optim as optim
from torch import nn


def to_np(tensor) -> np.ndarray:
    return tensor.cpu().detach().numpy()

class OptimizerCreator():

    def __init__(self):
        self.builders = {
            'Adam': lambda parameters, kwargs : optim.Adam(parameters, **kwargs)
        }

    def create(self, optimizer, parameters, kwargs):
        return self.builders[optimizer](parameters, kwargs)

class CriterionCreator():

    def __init__(self):
        self.builders = {
            'MSE': nn.MSELoss
        }

    def create(self, criterion):
        return self.builders[criterion]()

def get_files(path, key, day, month, year):

    files = [
    f for f in os.listdir(path) 
        if re.match(r'\d\d_\d\d_\d\d_.*', f)]
    
    if year: 
        files = [f for f in files if f[0:2]==year]
    if month: 
        files = [f for f in files if f[3:5]==month]
    if day: 
        files = [f for f in files if f[6:8]==day]
    if key: 
        files = [f for f in files if f[15:15+len(key)]==key]
    
    return files

def load_scores(path=None, day=None, month=None, year=None,
                key=None, display=False):
    
    if path is None:
        path = os.path.join(os.path.dirname(__file__), r'./../../output/score')
    files = get_files(path, key, day, month, year)
    
    if display:
        print(files)
        
    res = {}
    for f in files:
        score_key = f[15:].replace('.csv','')
        res[score_key] = pd.read_csv(os.path.join(path, f))
    return res

def plot_scores(dic_scores, window_size=20, target_score=None):
    
    fig, axe = plt.subplots(1,1,figsize=(12,6), dpi=175)

    max_len = 0
    for key, result in dic_scores.items():
        score = np.array(result.score)
        score_averaged = []
        for i in range(len(score)):
            score_averaged.append(
                np.mean(
                    score[max(0, i-window_size//2): min(len(score)-1, i+window_size//2)]))
        max_len = max(max_len, len(score_averaged))
        axe.plot(score_averaged, label=key)

    if target_score:
        axe.hlines(target_score, 0, max_len, 'k', linestyle=':', label='target score')

    axe.set_ylabel('Score')
    axe.set_xlabel('Episode #')
    fig.legend(bbox_to_anchor=(.985, .98), loc='upper left')
    plt.tight_layout()

def create_time_suffix():
    now = datetime.now()
    return f'{str(now.year)[-2:]}_{now.month:02}_{now.day:02}_{now.hour:02}h{now.minute:02}'

def save_scores(scores, scores_by_agent, key, path):

    df_scores = pd.DataFrame({
        f'score_{agent_name}' : score for agent_name, score in scores_by_agent.items()
    })
    df_scores['score'] = scores

    print(' ... saving score ...')
    df_scores.to_csv(os.path.join(path, f'{create_time_suffix()}_{key}.csv'), index=False)
    
def save_AC_models(MA_agent, key, path):

    file_name_base =f'{create_time_suffix()}_{key}'
    # absolute file path to avoir max 260 char bug.
    file_path = os.path.abspath(os.path.join(path, file_name_base))
    
    if not os.path.exists(file_path):
        os.makedirs(file_path)
        
    for agent_name, agent in MA_agent.agents_named.items():
        
        if hasattr(agent, "actor_network"):
            print('... saving actor ...')
            torch.save(agent.actor_network.state_dict(),
                       os.path.join(file_path, f'{file_name_base}_{agent_name}_actor.pth'))
            
        if hasattr(agent, "critic_network"):
            print('... saving critic ...')
            torch.save(agent.critic_network.state_dict(),
                       os.path.join(file_path, f'{file_name_base}_{agent_name}_critic.pth'))

def save_configuration(agent, key, path):
    
    config_file_name = os.path.join(path, f'{create_time_suffix()}_{key}.json')
    
    with open(config_file_name, 'w') as config_file:
        json.dump(agent.config.dict, config_file)

def filter_scores_on_averaged_threshold(dict_scores, th_score, window_size, sup=True):

    def has_at_least_mean_score(scores, th_score, window_size):
        return np.max((np.array([np.mean(scores[i:i+window_size]) for i in range(len(scores))]))) >= th_score 
    
    def has_at_most_mean_score(scores, th_score, window_size):
        return np.max((np.array([np.mean(scores[i:i+window_size]) for i in range(len(scores))]))) < th_score 
    
    if sup:
        func = has_at_least_mean_score
    else:
        func = has_at_most_mean_score
    
    res = {
        k:v for k,v in dict_scores.items() if func(np.array(v.score), th_score, window_size)
    }
    
    return res

import os
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from datetime import datetime
import json

import torch


def to_np(tensor) -> np.ndarray:
    return tensor.cpu().detach().numpy()

def load_scores(path=None, day=None, month=None, year=None,
                keys=None, display=False):
    '''
    Load and return the scores optionaly for a certain ime period in a dictionary.

    Args:
        path (str, optional): Path where the scores are stored.
        day (int, optional): The day of the runs to load.
        month (int, optional): The month of the runs to load.
        year (int, optional): The last two digits of the year of the runs to load.
        key (str or list[str], optional): Specific run keys to load.
        display (bool, optional): print the found score files.

    Returns:
        dict: pandas DataFrame of the scores by run key.
    '''
    if path is None:
        path = os.path.join(os.path.dirname(__file__), r'./../../output/score')
    keys = list(keys) if isinstance(keys, str) else keys

    files = get_files(path, keys, day, month, year)
    
    if display:
        print(files)
        
    res = {}
    for f in files:
        score_key = f[15:].replace('.csv','')
        res[score_key] = pd.read_csv(os.path.join(path, f))
    return res

def plot_scores(dic_scores, window_size=20, target_score=None, axe=None, colors=None, title=None):
    """
    Plot the global scores of the agents in function of the number of episodes.

    Args:
        dic_scores (dict): pandas DataFrame of the scores by run key.
        window_size (int, defaults to 20): The size of the window for the rolling average.
        target_score (float, optional): Display the given target score as a dotted horizontal line.
        axe (AxesSubplot, optional): axe of the subplot when this plot is embedded in a subplot.
        colors (list[mplt color], optional): list of color when default colors are not suitable.
        title (str, optional): title of the plot to display.
    """
    if axe:
        fig = plt.gcf()
    else:
        fig, axe = plt.subplots(1,1,figsize=(12,6), dpi=175)
    if not colors:
        colors = [None] * len(dic_scores)

    max_len = 0
    for idx,(key, result) in enumerate(dic_scores.items()):
        score = np.array(result.score)
        score_averaged = []
        for i in range(len(score)):
            score_averaged.append(
                np.mean(
                    score[max(0, i-window_size//2): min(len(score)-1, i+window_size//2)]))
        max_len = max(max_len, len(score_averaged))
        axe.plot(score_averaged, label=key, color=colors[idx])

    if target_score:
        axe.hlines(target_score, 0, max_len, 'k', linestyle=':', label='target score')

    axe.set_ylabel('Score')
    axe.set_xlabel('Episode #')
    if title:
        axe.set_title(title, fontdict={'fontsize': 14})
    fig.legend(bbox_to_anchor=(.985, .98), loc='upper left')
    plt.tight_layout()

def save_scores(scores, scores_by_agent, key, path):

    df_scores = pd.DataFrame({
        f'score_{agent_name}' : score for agent_name, score in scores_by_agent.items()
    })
    df_scores['score'] = scores

    print(' ... saving score ...')
    df_scores.to_csv(os.path.join(path, f'{_create_time_suffix()}_{key}.csv'), index=False)
    
def save_AC_models(MA_agent, key, path):

    file_name_base =f'{_create_time_suffix()}_{key}'
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
    
    config_file_name = os.path.join(path, f'{_create_time_suffix()}_{key}.json')
    
    with open(config_file_name, 'w') as config_file:
        json.dump(agent.config.dict, config_file)

def filter_scores_on_averaged_threshold(dict_scores, th_score, window_size, sup=True):
    """
    Filter a given dictionary of scores by its performances compared to a given threshold when
    averaged on a specific window size.

    Args:
        dict_scores (dict): dictionary of score to filter.
        th_score (float): threshold of score that will act as a lower or upped bound to filter.
        window_size (int): size of the rolling average to compute the performances.
        sup (bool, default to True): wheter the given threshold is an upper or lower bound.
    """

    def has_at_least_mean_score(scores, th_score, window_size):
        return np.any((np.array([np.mean(scores[i:i+window_size]) for i in range(len(scores))])) >= th_score) 
    
    def has_at_most_mean_score(scores, th_score, window_size):
        return np.any((np.array([np.mean(scores[i:i+window_size]) for i in range(len(scores))]))< th_score) 
    
    if sup:
        func = has_at_least_mean_score
    else:
        func = has_at_most_mean_score
    
    res = {
        k:v for k,v in dict_scores.items() if func(np.array(v.score), th_score, window_size)
    }
    
    return res

def get_files(path, keys, day, month, year):

    files = [
    f for f in os.listdir(path) 
        if re.match(r'\d\d_\d\d_\d\d_.*', f)]
    
    if year: 
        files = [f for f in files if f[0:2]==year]
    if month: 
        files = [f for f in files if f[3:5]==month]
    if day: 
        files = [f for f in files if f[6:8]==day]
    if keys: 
        files = [f for f in files if f[15: f.find('.')] in keys]
    
    return files

def _create_time_suffix():
    now = datetime.now()
    return f'{str(now.year)[-2:]}_{now.month:02}_{now.day:02}_{now.hour:02}h{now.minute:02}'
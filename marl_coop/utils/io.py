import os
import re
import json
from datetime import datetime

import pandas as pd
import torch


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
    keys = [keys] if isinstance(keys, str) else keys

    files = get_files(path, keys, day, month, year)
    
    if display:
        print(files)
        
    res = {}
    for f in files:
        score_key = f[15:].replace('.csv','')
        res[score_key] = pd.read_csv(os.path.join(path, f))
    return res

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
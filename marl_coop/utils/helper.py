import numpy as np

def to_np(tensor) -> np.ndarray:
    return tensor.cpu().detach().numpy()

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
import numpy as np
import matplotlib.pyplot as plt


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
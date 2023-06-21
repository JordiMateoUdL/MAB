from typing import List

from matplotlib import pyplot as plt
import numpy as np
from domain.bandit import Bandit
from mab.bandit_metrics import BanditMetrics
from mab.domain.algorithms.strategies import Strategy


def plot_strategies(bandit_class: Bandit, strategies: List[Strategy], 
                    num_rounds: int = 1000, xlabel: str = 'Rounds', 
                    ylabel: str = 'Average Reward', title: str = 'Comparison of Strategies',
                    plot_type: str = 'reward'):
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for strategy in strategies:
        bandit = bandit_class(strategy=strategy)
        _ = bandit.run(num_rounds=num_rounds)
        metrics = BanditMetrics(bandit)
        #print(bandit.rewards)
        if plot_type == 'reward':
            data = np.cumsum(bandit.rewards)
        elif plot_type == 'regret':
            data = np.cumsum(bandit.regrets)
        else:
            raise ValueError("Invalid plot_type. Choose between 'rewards' and 'regret'.")
        
        ax.plot(range(num_rounds), data, label=str(strategy))
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.set_title(title)
    
    plt.show()
    
    return fig
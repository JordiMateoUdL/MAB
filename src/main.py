from typing import List
from bandits.bandit_metrics import BanditMetrics
from bandits.helpers.plotter import plot_strategies
from bandits.strategies import EpsilonGreedy, PureExploration, PureExploitation, Strategy
from examples.slot_machine import SlotMachineBandit, SlotMachine
import matplotlib.pyplot as plt
import copy

if __name__ == "__main__":

    EXPLORATION_RATIO = 0.15    
    slot_machines_config =[
        {"true_reward": 1.5},
        {"true_reward": 2.0},
        {"true_reward": 3.0}
    ]
    
    slot_machines = []
    for config in slot_machines_config:
        slot_machines.append(SlotMachine(**config))
        
    strategies = [
        PureExploitation(arms=copy.deepcopy(slot_machines)),
        PureExploration(arms=copy.deepcopy(slot_machines)),
        EpsilonGreedy(arms=copy.deepcopy(slot_machines), epsilon=EXPLORATION_RATIO)
    ]
    
    plot_strategies(bandit_class=SlotMachineBandit,
                   strategies=strategies,
                   plot_type='regret',
                   ylabel='Cumulative Regret',
                   num_rounds=1000)
    
    # for strategy in strategies:
    #     bandit = SlotMachineBandit(strategy=strategy)
    #     best_arm = bandit.run(num_rounds=1000)
    
    #     metrics = BanditMetrics(bandit)
    #     metrics.plot_cumulative_rewards()
    #     metrics.plot_histogram_arm_selections()
    
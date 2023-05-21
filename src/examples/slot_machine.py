import numpy as np
from domain.arm import Arm
from bandits.bandit import Bandit
from bandits.strategies import Strategy

class SlotMachine(Arm):
    def __init__(self, true_reward):
        super().__init__()
        self.true_reward = true_reward

    def pull(self):
        # Based on uniform distribution
        reward = np.random.uniform(self.true_reward, 1)
        return reward
    
    def update_reward(self, reward):
         self.pull_counts += 1
         self.cumulative_reward = (self.cumulative_reward * 
                                   (self.pull_counts - 1) + reward) / self.pull_counts
    
class SlotMachineBandit(Bandit):
    def __init__(self, strategy:Strategy):
        super().__init__(strategy=strategy)
                    
    def _play_round(self):
        selected_arm = self.strategy.select_arm()
        reward = self.strategy.arms[selected_arm].pull()
        self.strategy.arms[selected_arm].update_reward(reward)
        return selected_arm
        
    def run(self, num_rounds):
        for _ in range(num_rounds):
            selected = self._play_round()
            self.rewards.append(self.strategy.arms[selected].cumulative_reward)
            self.regrets.append(self.strategy.arms[selected].get_average_regret(self.strategy.best_reward))
            #@TODO: Add logging
            #print(f"Round: {_} | Selected Arm: {selected} | Reward: {self.rewards[-1]} | Regret: {self.regrets[-1]}")
        
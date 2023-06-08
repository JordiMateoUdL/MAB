import numpy as np
from mab.domain.arm import Arm

class SlotMachine(Arm):
    """Class to represent a slot machine."""
    def __init__(self, true_reward):
        super().__init__()
        self._true_reward = true_reward

    def pull(self):
        # Based on uniform distribution
        reward = np.random.uniform(self._true_reward, 1)
        return reward

    def update_reward(self, reward):
        self._pull_counts += 1
        self._cumulative_reward = (self._cumulative_reward *
                                  (self._pull_counts - 1) + reward) / self._pull_counts

    #def get_reward(self):
    #    return self._true_reward
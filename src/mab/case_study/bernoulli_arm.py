"""Module for the BernoulliArm class."""
import random
from mab.domain.arm import Arm


class BernoulliArm(Arm):

    """Implementation of the BernoulliArm class for multi-armed bandit problems."""

    def __init__(self, success_probability: float = None) -> None:
        super().__init__()
        self.success_probability = success_probability
        if success_probability is None:
            self.success_probability = random.random()

    def pull(self) -> int | float:
        """Pull the arm based on the success probability and return the reward."""
        super().pull()

        if random.random() <= self.success_probability:
            return 1

        return 0

    def update_cumulative_reward(self, reward: int | float) -> None:
        """ We update the cumulative reward of the arm based on the pull count 
        and the reward obtained from pulling the arm."""
        self._cumulative_reward = (self._cumulative_reward * (self._pull_counts - 1)
                                   + reward) / self._pull_counts

    def __str__(self):
        return f"BernoulliArm p={self.success_probability}"

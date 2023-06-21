from typing import List
import random
from mab.domain.bandit_Old import Bandit
from mab.domain.arm import Arm


class EpsilonGreedyBandit(Bandit):
    """
    Bandit class implementing the epsilon-greedy strategy.
    """

    def __init__(self, arms: List[Arm], epsilon: float) -> None:
        super().__init__(arms)
        self.epsilon = epsilon

    def select_arm(self) -> Arm:

        if random.random() < self.epsilon:
            # Explore: Select a random arm
            return random.choice(self._arms)

        # Exploit: Select the arm with the highest average reward
        return max(self._arms, key=lambda arm: arm.get_average_reward())

    def update(self, arm: Arm, reward: float) -> None:
        super().update(arm, reward)
        # Update the average reward estimate for the selected arm
        arm.update_average_reward(reward)

    def __str__(self) -> str:
        """
        Returns:
            A string representing the EpsilonGreedy strategy.
        """
        return f'EpsilonGreedy(e={self.epsilon})'

"""
    Module: arm.py
    Abstract base class representing an arm in a multi-armed bandit problem.

    This module defines the abstract base class `Arm` that serves as a blueprint for implementing
    different arms in a multi-armed bandit problem. Each arm represents a possible action or choice
    that can be taken, and the goal is to find the arm that maximizes the cumulative reward.

    Subclasses of `Arm` must implement the abstract methods `pull` and `update_reward` according to
    their specific implementation details.

    Example usage:
        class MyArm(Arm):
            def pull(self):
                # Implement the logic to pull the arm and obtain the reward

            def update_reward(self, reward):
                # Implement the logic to update the reward estimate based on the obtained reward

        my_arm = MyArm()
        reward = my_arm.pull()
        my_arm.update_reward(reward)
        regret = my_arm.get_regret(best_reward)
        arm.reset()
"""

from abc import ABC, abstractmethod
from typing import Union

class Arm(ABC):
    """
    Abstract base class representing an arm in a multi-armed bandit problem.
    """

    def __init__(self) -> None:
        """
        Initializes the arm object.

        Initializes the pull_counts and cumulative_reward variables.
        """
        self.pull_counts:int = 0 
        self.cumulative_reward:Union[int, float] = 0 #cumulative reward obtained from pulling the arm

    @abstractmethod
    def pull(self) -> Union[int, float]:
        """
        Abstract method representing pulling the arm.

        Subclasses must implement this method with their specific implementation details.

        Returns:
            The reward obtained from pulling the arm.
        """
        raise NotImplementedError("pull method must be implemented...")

    @abstractmethod
    def update_reward(self, reward: Union[int, float]) -> None:
        """
        Abstract method for updating the arm's reward estimate.

        Args:
            reward: The reward obtained by pulling the arm.

        Subclasses must implement this method with their specific implementation details.
        """
        raise NotImplementedError("update_reward method must be implemented...")

    def reset(self) -> None:
        """
        Resets the arm's state.

        Resets the pull_counts and cumulative_reward attributes to their initial values.
        """
        self.pull_counts = 0
        self.cumulative_reward = 0
    
    def get_average_regret(self, best_reward: Union[int, float]) -> Union[int, float]:
        """
        Calculates the average regret over time for the arm.

        Args:
            best_reward: The best possible reward achievable obtained from any of the arms in the bandit.

        Returns:
            The regret of the arm, which is the difference between the best_reward 
            and the cumulative reward for this arm.

        Raises:
            ValueError: If the best_reward is negative.
        """
        if best_reward < 0:
            raise ValueError("The best_reward cannot be negative.")

        return best_reward - self.cumulative_reward

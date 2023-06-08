"""
    Module: arm.py
    Abstract base class representing an arm in a multi-armed bandit problem.

    This module defines the abstract base class `Arm` that serves as a blueprint for implementing
    different arms in a multi-armed bandit problem. Each arm represents a possible action or choice
    that can be taken, and the goal is to find the arm that maximizes the cumulative reward.
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
        # number of times the arm has been pulled
        self._pull_counts: int = 0
        # cumulative reward obtained from pulling the arm
        self._cumulative_reward: Union[int, float] = 0
        self._cumulative_regret: Union[int, float] = 0
        

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
        raise NotImplementedError(
            "update_reward method must be implemented...")
        
    # @abstractmethod
    # def get_reward(self) -> Union[int, float]:
    #     """
    #     Abstract method for getting the arm's reward estimate.

    #     Subclasses must implement this method with their specific implementation details.
    #     """
    #     raise NotImplementedError(
    #         "get_reward method must be implemented...")

    def reset(self) -> None:
        """
        Resets the arm's state.

        Resets the pull_counts and cumulative_reward attributes to their initial values.
        """
        self._pull_counts = 0
        self._cumulative_reward = 0
        self._cumulative_regret = 0

    def get_pull_counts(self) -> int:
        """
        Returns the number of times the arm has been pulled.

        Returns:
            The number of times the arm has been pulled.
        """
        return self._pull_counts

    def get_cumulative_reward(self) -> Union[int, float]:
        """
        Returns the cumulative reward obtained from pulling the arm.

        Returns:
            The cumulative reward obtained from pulling the arm.
        """
        return self._cumulative_reward
    

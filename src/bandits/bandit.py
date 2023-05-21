"""
Module: bandit.py
Abstract base class representing a bandit algorithm.

This module defines the abstract base class `Bandit` that serves as a blueprint for implementing
different bandit algorithms. 

Subclasses of `Bandit` must implement the abstract methods `play_round` and `run` according to
their specific implementation details.

Example usage:
    class MyBandit(Bandit):
        def play_round(self):
             # Implement the logic for a single round of the bandit algorithm

        def run(self, num_rounds):
            # Implement the main execution loop of the bandit algorithm

    strategy = MyStrategy()  # Instantiate your strategy here
    my_bandit = MyBandit(strategy)
    my_bandit.run(1000)
"""

from abc import ABC, abstractmethod
from typing import List

from bandits.strategies import Strategy

class Bandit(ABC):
    """
    Abstract base class representing a bandit algorithm.

    This class defines the interface for implementing different bandit algorithms.
    Subclasses must provide implementations for the abstract methods `play_round` and `run`.

    Attributes:
        strategy (Strategy): The strategy used by the bandit algorithm.
        rewards (List[float]): List to keep track of cumulative rewards across all arms and time.
        regrets (List[float]): List to keep track of cumulative regrets across all arms and time.
    """

    def __init__(self, strategy: Strategy) -> None:
        """
        Initializes the Bandit object.

        Args:
            strategy (Strategy): The strategy used by the bandit algorithm.
        """
        self.strategy = strategy
        self.rewards: List[float] = []
        self.regrets: List[float] = []

    @abstractmethod
    def _play_round(self) -> None:
        """
        Abstract method representing a single round of the bandit algorithm.

        Subclasses must implement this method with their specific implementation details.
        """
        raise NotImplementedError("play_round method must be implemented...")

    @abstractmethod
    def run(self, num_rounds: int = 1000) -> None:
        """
        Abstract method representing the main execution loop of the bandit algorithm.

        Args:
            num_rounds (int): The number of rounds to run the bandit algorithm. Default is 1000.
        """
        raise NotImplementedError("run method must be implemented...")

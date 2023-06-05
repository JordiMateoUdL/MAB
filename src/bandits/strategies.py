""" Strategy classes for selecting an arm in a multi-armed bandit problem. """
from abc import ABC, abstractmethod
from typing import List, Union
import numpy as np

from domain.arm import Arm


class Strategy(ABC):

    """
    Abstract base class representing a strategy for selecting 
    an arm in a multi-armed bandit problem.
    """

    def __init__(self, arms: List[Arm]):
        self.arms: List[Arm] = arms
        self.num_arms: int = len(arms)
        self.best_reward: Union(int, float) = np.max(
            [arm.true_reward for arm in arms])
        self.exploration_steps: int = 0
        self.exploitation_steps: int = 0

    @abstractmethod
    def select_arm(self) -> int:
        """
        Abstract method for selecting an arm based on the strategy.

        Returns:
            The index of the selected arm.
        """

        raise NotImplementedError("select_arm method must be implemented...")
    


class EpsilonGreedy(Strategy):

    """
    Epsilon-Greedy strategy for selecting an arm
    """

    def __init__(self, arms: List[Arm], epsilon: float = 0.1):
        """
        Initializes the EpsilonGreedy strategy.

        Args:
            arms (List[Arm]): The list of arms available.
            epsilon (float): The exploration rate (0 to 1) 
            for balancing exploration and exploitation.
        """
        super().__init__(arms=arms)
        self.epsilon: float = epsilon

    def select_arm(self) -> int:
        """
        Selects an arm based on the Epsilon-Greedy strategy.

        Returns:
            The index of the selected arm.
        """
        if np.random.random() < self.epsilon:
            self.exploration_steps += 1
            return np.random.randint(0, self.num_arms)

        self.exploitation_steps += 1
        return np.argmax([arm.cumulative_reward for arm in self.arms])

    def __str__(self) -> str:
        """
        Returns:
            A string representing the EpsilonGreedy strategy.
        """
        return f'EpsilonGreedy(e={self.epsilon})'


class UCB1(Strategy):
    """UCB1 (Upper Confidence Bound 1) strategy for selecting an arm"""

    def __init__(self, arms: List[Arm], c: Union[int, float] = 1.0, threshold: float = 1e-8):
        """
        Initializes the UCB1 strategy.

        Args:
            arms (List[Arm]): The list of arms available.
            c (Union[int,float]): The exploration parameter.
            threshold (float): A small value added to avoid division 
            by zero when calculating exploration term.
        """
        super().__init__(arms=arms)
        self.t_value: int = 0
        self.c_value: Union[int, float] = c
        self.threshold: float = threshold

    def __str__(self) -> str:
        """
        Returns a string representation of the UCB1 strategy.

        Returns:
            str: The string representation of the UCB1 strategy.
        """
        return f'UCB1(c={self.c_value})'

    def select_arm(self) -> int:
        """
        Selects the arm based on the UCB1 strategy.

        Returns:
            int: The index of the selected arm.
        """
        selected = np.argmax(
            [arm.cumulative_reward
             + (self.c_value * np.sqrt(np.log(self.t_value) / (arm.pull_counts + self.threshold)))
             if arm.pull_counts > 0 else float('inf') for arm in self.arms])

        self.t_value += 1
        return selected


class ThomsonSampling(Strategy):
    """
    Thompson Sampling strategy for the multi-armed bandit problem.
    """

    def __init__(self, arms: List[Arm], threshold: Union[int, float] = 0.5):
        """
        Initializes the Thomson Sampling strategy.

        Args:
            arms (List[Arm]): The list of arms available.
            threshold (Union[int, float]): The threshold for considering a reward as a success.
        """
        super().__init__(arms=arms)
        self.successes: List[int] = [1]*len(arms)
        self.failures: List[int] = [1]*len(arms)
        self.threshold: Union[int, float] = threshold

    def __str__(self) -> str:
        """
        Returns a string representation of the strategy.
        """
        return f'ThomsonSampling (threshold={self.threshold})'

    def select_arm(self) -> int:
        """
        Selects an arm to play based on the Thompson Sampling strategy.

        Returns:
            int: The index of the selected arm.
        """
        selected = np.argmax([np.random.beta(
            self.successes[arm], self.failures[arm]) for arm in range(len(self.arms))])

        return selected

    def update(self, reward: float, arm: int):
        """
        Updates the success and failure counts based on the received reward.

        Args:
            reward (float): The reward obtained from playing the selected arm.
            arm (int): The index of the selected arm.
        """
        if reward > self.threshold:
            self.successes[arm] += 1
        else:
            self.failures[arm] += 1


class PureExploration(Strategy):

    """
    Pure exploration strategy for selecting an arm. Always selects a random arm.
    """

    def __init__(self, arms: List[Arm]):
        """
        Initializes the PureExploration strategy.

        Args:
            arms (List[Arm]): The list of arms available in the bandit problem.
        """
        super().__init__(arms=arms)

    def select_arm(self):
        """
        Selects an arm based on the PureExploration strategy.

        Returns:
            The index of the selected arm.
        """
        self.exploration_steps += 1
        return np.random.randint(0, self.num_arms)

    def __str__(self) -> str:
        """
        Returns:
            A string representing the PureExploration strategy.
        """
        return "PureExploration"


class PureExploitation(Strategy):
    """
    Pure exploitation strategy for selecting an arm. Always selects the arm with the highest cumulative reward.
    """

    def __init__(self, arms: List[Arm]):
        """
        Initializes the PureExploration strategy.

        Args:
            arms (List[Arm]): The list of arms available.
        """
        super().__init__(arms=arms)

    def select_arm(self):
        """
        Selects an arm based on the PureExploration strategy.
        Returns:
            The index of the selected arm.
        """
        self.exploitation_steps += 1
        return np.argmax([arm.cumulative_reward for arm in self.arms])

    def __str__(self) -> str:
        """
        Returns:
            A string representing the PureExploration strategy.
        """
        return "PureExploitation"

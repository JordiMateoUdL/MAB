from abc import ABC, abstractmethod
from typing import List
import matplotlib.pyplot as plt

from mab.domain.arm import Arm

class ArmMetric(ABC):
    """
    Abstract base class representing an arm metric for visualization.
    """

    def __init__(self):
        pass

    @abstractmethod
    def compute_metric(self, arms: List['Arm']) -> List[float]:
        """
        Abstract method to compute the metric values for each arm.

        Args:
            arms: A list of Arm objects.

        Returns:
            A list of metric values for each arm.
        """
        raise NotImplementedError("compute_metric method must be implemented...")

    @abstractmethod
    def plot_metric(self, arms: List['Arm']) -> None:
        """
        Plots the metric values for each arm.

        Args:
            arms: A list of Arm objects.
        """
        raise NotImplementedError("plot_metric method must be implemented...")

    def __call__(self, arms: List['Arm']) -> None:
        """
        Calls the plot_metric method when the object is called as a function.

        Args:
            arms: A list of Arm objects.
        """
        self.plot_metric(arms)


class CumulativeRewardMetric(ArmMetric):
    """
    Arm metric representing the cumulative reward of each arm over time.
    """
    
    def __init__(self):
        super().__init__()
        self.name = "cumulative-reward"

    def compute_metric(self, arms: List['Arm']) -> List[float]:
        """
        Computes the cumulative reward of each arm over time.

        Args:
            arms: A list of Arm objects.

        Returns:
            A list of cumulative reward values for each arm.
        """
        cumulative_rewards = [arm.get_cumulative_reward() for arm in arms]
        return cumulative_rewards

    def plot_metric(self, arms: List['Arm']) -> None:
        """
        Plots the cumulative reward of each arm over time.

        Args:
            arms: A list of Arm objects.
        """
        cumulative_rewards = self.compute_metric(arms)
        time_steps = list(range(len(cumulative_rewards)))
        print(cumulative_rewards)
        plt.figure()
        for arm_idx in enumerate(arms):
            plt.plot(time_steps, cumulative_rewards[arm_idx], label=f"Arm {arm_idx + 1}")

        plt.xlabel("Time Steps")
        plt.ylabel("Cumulative Reward")
        plt.legend()
        plt.title("Cumulative Reward of Arms over Time")
        plt.show()
         
    def __str__(self):
        return "Cumulative Reward Metric"


class RegretMetric(ArmMetric):
    """
    Arm metric representing the regret of each arm over time.
    """

    def __init__(self, best_reward: float):
        super().__init__()
        self.best_reward = best_reward

    def compute_metric(self, arms: List['Arm']) -> List[float]:
        """
        Computes the regret of each arm over time.

        Args:
            arms: A list of Arm objects.

        Returns:
            A list of regret values for each arm.
        """
        regrets = [self.best_reward - arm.cumulative_reward for arm in arms]
        return regrets

    def plot_metric(self, arms: List['Arm']) -> None:
        """
        Plots the regret of each arm over time.

        Args:
            arms: A list of Arm objects.
        """
        regrets = self.compute_metric(arms)
        time_steps = list(range(len(regrets)))

        plt.figure()
        for arm_idx in enumerate(arms):
            plt.plot(time_steps, regrets[arm_idx], label=f"Arm {arm_idx + 1}")

        plt.xlabel("Time Steps")
        plt.ylabel("Regret")
        plt.legend()
        plt.title("Regret of Arms over Time")
        plt.show()

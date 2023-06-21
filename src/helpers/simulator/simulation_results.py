"""Package to store the results of a simulation run"""

from typing import List

class SimulationResults:
    """
    Class to store and analyze the metrics generated during a multi-armed bandit simulation.
    """

    def __init__(self) -> None:
        self.cumulative_regret: List[float] = []
        self.instantaneous_regret: List[float] = []
        self.cumulative_reward: List[float] = []
        self.average_regret: List[float] = []
        self.average_reward: List[float] = []
        self.exploration_steps: int = 0
        self.exploitation_steps: int = 0
        self.arm_pulls: List[int] = []

    def update_regret(self, cumulative_regret: float, instantaneous_regret: float) -> None:
        """
        Update the cumulative and instantaneous regret at each time step.

        Args:
            cumulative_regret: The cumulative regret at the current time step.
            instantaneous_regret: The instantaneous regret at the current time step.
        """
        self.cumulative_regret.append(cumulative_regret)
        self.instantaneous_regret.append(instantaneous_regret)

    def update_reward(self, cumulative_reward: float, average_reward: float) -> None:
        """
        Update the cumulative and average reward at each time step.

        Args:
            cumulative_reward: The cumulative reward at the current time step.
            average_reward: The average reward at the current time step.
        """
        self.cumulative_reward.append(cumulative_reward)
        self.average_reward.append(average_reward)

    def update_exploration_steps(self) -> None:
        """
        Increment the count of exploration steps.
        """
        self.exploration_steps += 1

    def update_exploitation_steps(self) -> None:
        """
        Increment the count of exploitation steps.
        """
        self.exploitation_steps += 1

    def update_arm_pulls(self, arm_index: int, num_pulls: int) -> None:
        """
        Update the number of pulls for a specific arm.

        Args:
            arm_index: The index of the arm.
            num_pulls: The number of pulls for the arm at the current time step.
        """
        if len(self.arm_pulls) <= arm_index:
            self.arm_pulls.extend([0] * (arm_index + 1 - len(self.arm_pulls)))
        self.arm_pulls[arm_index] = num_pulls
        
        
    def get_cumulative_rewards(self) -> List[float]:
            """
            Returns the list of cumulative rewards at each time step.

            Returns:
                A list of cumulative rewards at each time step.
            """
            return self.cumulative_reward
        
    def get_cumulative_regrets(self) -> List[float]:
            """
            Returns the list of cumulative regrets at each time step.

            Returns:
                A list of cumulative regrets at each time step.
            """
            return self.cumulative_regret   
        
        
    def get_cumulative_reward(self) -> float:
            """
            Returns the cumulative reward at the final time step.

            Returns:
                The cumulative reward at the final time step.
            """
            return self.cumulative_reward[-1]
        
    def get_average_reward(self) -> float:
            """
            Returns the average reward at the final time step.

            Returns:
                The average reward at the final time step.
            """
            return self.average_reward[-1]
        
    def get_cumulative_regret(self) -> float:
            """
            Returns the cumulative regret at the final time step.

            Returns:
                The cumulative regret at the final time step.
            """
            return self.cumulative_regret[-1]
        
    def get_exploration_steps(self) -> int:
            """
            Returns the number of exploration steps.

            Returns:
                The number of exploration steps.
            """
            return self.exploration_steps
        
    def get_exploitation_steps(self) -> int:
            """
            Returns the number of exploitation steps.

            Returns:
                The number of exploitation steps.
            """
            return self.exploitation_steps
        
    def get_arm_pulls(self) -> List[int]:
            """
            Returns the list of arm pulls.

            Returns:
                A list of arm pulls.
            """
            return self.arm_pulls
        
      


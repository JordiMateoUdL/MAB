
from typing import List, Union
from mab.domain.arm import Arm

class Bandit():

    def __init__(self, arms: List[Arm]) -> None:
        """
        Initializes the bandit object.

        Args:
            arms: A list of Arm objects representing the arms in the bandit.
        """
        self._arms = arms
        self._cumulative_reward: Union[int, float] = 0
        self._cumulative_regret: Union[int, float] = 0


    def update(self, arm: Arm, reward: float) -> None:
        """
        Updates the bandit's internal state after pulling an arm and receiving a reward.

        Args:
            arm: The Arm object that was pulled.
            reward: The reward obtained from pulling the arm.
        """
        # Update the arm's internal state
        arm.update_reward(reward)
        
        # Update bandit's internal state
        self.update_cumulative_reward(reward)
        #regret = self.get_regret(reward)
        #self.update_cumulative_regret(regret)

    def reset(self) -> None:
        """
        Resets the bandit's state.

        Resets the state of all arms in the bandit.
        """
        for arm in self._arms:
            arm.reset()
        self._cumulative_reward = 0
        self._cumulative_regret = 0
              
    def get_arms(self) -> List[Arm]:
        """
        Returns the list of arms in the bandit.

        Returns:
            A list of Arm objects.
        """
        return self._arms
    
    def get_total_arms(self) -> int:
        """
        Returns the total number of arms in the bandit.

        Returns:
            The total number of arms in the bandit.
        """
        return len(self._arms)
    
    def get_arm_index(self, arm:Arm) -> int:
        """
        Returns the index of the arm in the bandit.

        Returns:
            The index of the arm in the bandit.
        """
        return self._arms.index(arm)

    def get_best_reward(self) -> float:
        """
        Returns the best possible reward in the bandit.

        Returns:
            The best possible reward in the bandit.
        """
        return max([arm.get_reward() for arm in self._arms])
    
    def get_regret(self, reward: float) -> float:
        """
        Calculates the regret of the bandit.

        Args:
            reward: The reward obtained from pulling the arm.

        Returns:
            The regret of the bandit.
        """
        return self.get_best_reward() - reward
    
    def update_cumulative_reward(self, reward: float) -> None:
        """
        Updates the cumulative reward of the bandit.

        Args:
            reward: The reward obtained from pulling the arm.
        """
        self._cumulative_reward += reward
        
    def update_cumulative_regret(self, regret: float) -> None:
        """
        Updates the cumulative regret of the bandit.

        Args:
            regret: The regret of the bandit.
        """
        self._cumulative_regret += regret  
        
    def get_cumulative_reward(self) -> float:
        """
        Returns the cumulative reward of the bandit.

        Returns:
            The cumulative reward of the bandit.
        """
        return self._cumulative_reward
    
    def get_cumulative_regret(self) -> float:
        """
        Returns the cumulative regret of the bandit.

        Returns:
            The cumulative regret of the bandit.
        """
        return self._cumulative_regret  
    


    
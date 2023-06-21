'''
Module: bandit.py
Base class representing a multi-armed bandit problem.
'''
from typing import List
from mab.domain.arm import Arm

class Bandit:
    """
    Class representing a multi-armed bandit problem.

    This class represents a bandit consisting of a set of arms, each representing a possible action
    or choice that can be taken, and the goal is to find the arm that maximizes the cumulative reward.
    """

    def __init__(self, arms: List[Arm] = None) -> None:
        """
        Initializes the bandit object.

        Initializes the arms variable with the provided list of arms.
        If no list is provided, it initializes it as an empty list.
        """
        self._arms: List[Arm] = arms or []

    def get_arms(self) -> List[Arm]:
        """
        Returns:
            A list of arms representing the possible actions or choices.
        """
        return self._arms

    def add_arm(self, arm: Arm) -> None:
        """
        Adds an arm to the bandit.

        Args:
            arm: The arm to be added to the bandit.
        """
        self._arms.append(arm)

    def remove_arm(self, arm: Arm) -> None:
        """
        Removes an arm from the bandit.

        Args:
            arm: The arm to be removed from the bandit.
        """
        self._arms.remove(arm)
        
    def get_arm(self, index: int) -> Arm:
        """
        Returns the arm at the specified index.
        Args: The index of the arm to be returned.
        """
        return self._arms[index]
    
    def get_arm_index(self, arm:Arm) -> int:
        """
        Returns the index of the arm in the bandit.

        Returns:
            The index of the arm in the bandit.
        """
        return self._arms.index(arm)

    def get_arms_number(self) -> int:
        """
        Returns the total number of arms in the bandit.

        Returns:
            The total number of arms in the bandit.
        """
        return len(self._arms)

    def calculate_cumulative_reward(self) -> float:
        """
        Calculates the cumulative reward of the bandit.

        Returns:
            The cumulative reward obtained from all the arms in the bandit.
        """
        return sum(arm.get_cumulative_reward() for arm in self._arms)

    def reset(self) -> None:
        """
        Resets the bandit's state.

        Resets the state of all arms in the bandit.
        """
        for arm in self._arms:
            arm.reset()
            
    def pull_arm(self, arm: Arm) -> float:
        """
        Pulls an arm from the bandit.

        Args:
            arm: The arm to be pulled.

        Returns:
            The reward obtained from pulling the arm.
        """
        
        if arm in self._arms:
            reward = arm.pull()
            arm.update_cumulative_reward(reward)
            return reward
        else:
            raise ValueError("The arm is not in the bandit.")

    def __clone__(self):
        """
        Creates a new instance of the bandit with the same arms.

        Returns:
            A new instance of the bandit with the same arms.
        """
        cloned_bandit = self.__class__()
        cloned_bandit._arms = [arm.__clone__() for arm in self._arms]
        return cloned_bandit
'''
This module defines the abstract base class `Solver` that serves as a blueprint 
for implementing different solvers for Bandit problems.
'''

import random
from enum import Enum
from typing import Optional
from abc import ABC, abstractmethod
from mab.domain.arm import Arm

from mab.domain.bandit_Old import Bandit

class SolverAction(Enum):
    '''
    This enum represents the different actions that a solver can take.
    '''
    EXPLORE = "Explore"
    EXPLOIT = "Exploit"
    
class Solver(ABC):
    '''
    Abstract base class representing a solver for a multi-armed bandit problem.
    '''
    def __init__(self, bandit: Optional[Bandit] = None) -> None:
        self._bandit = bandit
        self.rewards = {}
        self.counts = {}
        self._action = None

    @abstractmethod
    def select_arm(self) -> Arm:
        '''
        This is an abstract method that must be implemented by subclasses.
        The aim of this method is to select an arm from the bandit.
        '''
        
        raise NotImplementedError("select_arm method must be implemented...")
   

    def get_bandit(self) -> Bandit:
        
        return self._bandit
    
    def update_bandit(self, arm: Arm, reward: float) -> None:
        self._bandit.update(arm, reward)
        
    def get_action(self) -> BanditAction:
        return self._action
        
class EpsilonGreedy(BanditSolver):
    def __init__(self, bandit, epsilon):
        super().__init__(bandit)
        self._epsilon = epsilon

    def select_arm(self) -> Arm:

        arms = self._bandit.get_arms()

        if random.random() < self._epsilon:
            # Explore: Select a random arm
            self._action = BanditAction.EXPLORATION
            return random.choice(arms)

        # Exploit: Select the arm with the highest average reward
        self._action = BanditAction.EXPLOITATION
        return max(arms, key=lambda arm: arm.get_cumulative_reward())
    
    def __str__(self) -> str:
        return f" EpsilonGreedy(epsilon={self._epsilon})"
    

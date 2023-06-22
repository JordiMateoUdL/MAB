'''
This module contains the abstract base class representing a solver for a multi-armed bandit problem.
'''

from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Tuple
from mab.domain.arm import Arm

from mab.domain.bandit import Bandit


class SolverAction(Enum):
    '''
    This enum represents the different actions that a solver can take.
    '''
    EXPLORE = "Explore"
    EXPLOIT = "Exploit"


class Solver(ABC):
    '''Abstract base class representing a solver for a multi-armed bandit problem'''

    def __init__(self, bandit: Bandit) -> None:
        """ 
        Initializes the solver with a bandit problem.
        Uses injection pattern to inject the bandit problem into the solver.
        Args:
            bandit (Bandit): The multi-armed bandit problem to solve.
        """
        self._bandit = bandit
        self._action_history: List[Tuple[Arm, SolverAction]] = []

    @abstractmethod
    def select_arm(self) -> int:
        """
        Selects an arm from the bandit to pull.

        Returns:
            The index of the selected arm.
        """

        raise NotImplementedError("select_arm method must be implemented...")

    def update_solver_history(self, arm: Arm, action: SolverAction) -> None:
        """
        Updates the solver's history with the arm, action, and reward.

        Args:
            arm (Arm): The arm that was pulled.
            action (SolverAction): The action that was taken.
        """

        self._action_history.append((arm, action))

    def get_action_history(self) -> List[Tuple[Arm, SolverAction]]:
        """
        Returns the solver's history.

        Returns:
            The solver's history.
        """

        return self._action_history

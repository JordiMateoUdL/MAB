"""Module for defining Upper confidence bound based solvers."""

from math import log, sqrt
from mab.domain.arm import Arm
from mab.domain.bandit import Bandit
from mab.domain.solver import Solver, SolverAction


class UCB1Solver(Solver):
    """
    Solver implementing the UCB1 algorithm for multi-armed bandit problems.

    The UCB1 algorithm balances exploration and exploitation by selecting arms 
    based on their upper confidence bounds.

    Args:
        bandit (Bandit): The multi-armed bandit problem to solve.
        exploration_parameter (float): The exploration parameter (c) for UCB1.

    Attributes:
        bandit (Bandit): The multi-armed bandit problem to solve.
        exploration_parameter (float): The exploration parameter (c) for UCB1.
    """

    def __init__(self, bandit: Bandit, exploration_parameter: float) -> None:
        super().__init__(bandit)
        self.exploration_parameter = exploration_parameter

    def select_arm(self) -> Arm:
        """
        Selects an arm from the bandit to pull using the UCB1 algorithm.

        Returns:
            The selected arm.
        """
        arms = self._bandit.get_arms()
        total_pulls = sum(arm.get_pull_counts() for arm in arms)

        if total_pulls == 0:
            return arms[0]  # Select the first arm if no pulls have been made

        ucb_values = []
        for arm in arms:
            exploration_term = sqrt(
                (2 * log(total_pulls)) / max(1, arm.get_pull_counts()))
            exploration_bonus = exploration_term * self.exploration_parameter
            ucb_value = arm.get_cumulative_reward() + exploration_bonus
            ucb_values.append(ucb_value)

        max_ucb_index = ucb_values.index(max(ucb_values))
        selected_arm = arms[max_ucb_index]

        if exploration_bonus > 0:
            self.update_solver_history(selected_arm, SolverAction.EXPLORE)
        else:
            self.update_solver_history(selected_arm, SolverAction.EXPLOIT)

        return selected_arm

    def __str__(self):
        """Returns the name of the solver."""
        return f'UCB1(c={self.exploration_parameter})'

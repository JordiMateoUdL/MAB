"""Module for the EpsilonGreedy Solvers class."""""
import random
from mab.domain.bandit import Bandit
from mab.domain.arm import Arm
from mab.domain.solver import Solver, SolverAction


class EpsilonGreedySolver(Solver):
    """
    Solver implementing the epsilon-greedy algorithm for multi-armed bandit problems.

    The epsilon-greedy algorithm balances exploration and exploitation 
    by choosing a random arm with probability epsilon and choosing the
    arm with the highest cumulative reward with probability 1 - epsilon.

    Args:
        bandit (Bandit): The multi-armed bandit problem to solve.
        epsilon (float): The exploration parameter. Should be a value between 0 and 1.

    Attributes:
        bandit (Bandit): The multi-armed bandit problem to solve.
        epsilon (float): The exploration parameter.

    """

    def __init__(self, bandit: Bandit, epsilon: float) -> None:
        super().__init__(bandit)
        self.epsilon = epsilon

    def select_arm(self) -> Arm:
        """
        Selects an arm using the epsilon-greedy algorithm.

        Returns:
            The selected arm.
        """
        if random.random() > self.epsilon:
            return self.exploit()

        return self.explore()

    def exploit(self) -> Arm:
        """
        Exploits the arm with the highest cumulative reward.

        Returns:
            The selected arm for exploitation.
        """
        arms = self._bandit.get_arms()
        selected_arm = max(arms, key=lambda arm: arm.get_cumulative_reward())
        self.update_solver_history(selected_arm, SolverAction.EXPLOIT)
        return selected_arm

    def explore(self) -> Arm:
        """
        Explores a random arm.

        Returns:
            The randomly selected arm for exploration.
        """
        arms = self._bandit.get_arms()
        selected_arm = random.choice(arms)
        self.update_solver_history(selected_arm, SolverAction.EXPLORE)
        return selected_arm

    def __str__(self) -> str:
        return f"EpsilonGreedy(epsilon={self.epsilon})"

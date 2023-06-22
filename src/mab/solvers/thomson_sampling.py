"""Module for defining ThomsonSampling based solvers."""

from random import betavariate, choice
from mab.domain.arm import Arm
from mab.domain.bandit import Bandit
from mab.domain.solver import Solver, SolverAction


class ThomsonSamplingSolver(Solver):
    """Thomson Sampling Solver implementation for multi-armed bandit problems."""

    def __init__(self, bandit: Bandit, 
                 exploration_parameter: float = 0.0,
                 init_a: float = 1,
                 init_b: float = 1) -> None:
        """
        Initialize the ThomsonSamplingSolver.

        Args:
            exploration_parameter (float): The exploration parameter controlling the balance
                between exploration and exploitation. A higher value encourages more exploration.
            init_a (float): The initial value of the alpha parameter of the Beta distribution.
            init_b (float): The initial value of the beta parameter of the Beta distribution.

        """
        super().__init__(bandit)
        self.exploration_parameter = exploration_parameter
        self._alpha = [init_a] * bandit.get_arms_number()
        self._beta = [init_b] * bandit.get_arms_number()

    def select_arm(self) -> Arm:
        """
        Selects an arm from the bandit using the Thomson Sampling algorithm.

        Returns:
            The selected arm.

        """
        samples = [betavariate(self._alpha[i], self._beta[i]) 
                   for i in range(self._bandit.get_arms_number())]
        max_sample = max(samples)
        max_indices = [i for i, sample in enumerate(samples) if sample == max_sample]
        selected_arm_index = choice(max_indices)
        return self._bandit.get_arm(selected_arm_index)
    
    def update_state(self, arm: Arm, reward: float) -> None:
        """Updates the state of the solver based on the reward obtained from pulling the arm."""
        arm_index = self._bandit.get_arm_index(arm)
        self._alpha[arm_index] += reward
        self._beta[arm_index] += (1 - reward)

        

    def __str__(self):
        """Returns the name of the solver."""
        return f'Thomson Sampling(c={self.exploration_parameter})'

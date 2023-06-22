"""Module for running simulations."""
from typing import List
from mab.domain.bandit import Bandit
from mab.domain.solver import Solver
from mab.solvers.thomson_sampling import ThomsonSamplingSolver


class SimulationResults:
    """Helper class to store the results of the simulation."""

    def __init__(self):
        self.rewards = []  # The rewards for each iteration
        self.actions = []  # The actions for each iteration
        self.usage_fractions = {}  # The fraction of times each arm was selected
        self.cummulatives = []  # The cumulative rewards for each arm

    def get_rewards(self) -> List[float]:
        """Returns the rewards for each iteration."""
        return self.rewards

    def get_actions(self) -> List[float]:
        """Returns the actions for each iteration."""
        return self.actions


class Simulator:
    '''Class representing a simulator for a multi-armed bandit problem.'''

    def __init__(self, bandit: Bandit, solvers: List[Solver]) -> None:
        self.bandit = bandit
        self.solvers = solvers
        self.results = {solver: SimulationResults() for solver in solvers}

    def run(self, num_iterations: int) -> None:
        '''Runs the simulation for the specified number of iterations.'''
        for solver in self.solvers:
            for _ in range(num_iterations):
                selected_arm = solver.select_arm()
                if selected_arm not in self.bandit.get_arms():
                    raise ValueError(
                        "Selected arm is not present in the bandit.")
                reward = self.bandit.pull_arm(selected_arm)
                solver.update_solver_history(selected_arm, reward)
                self.results[solver].rewards.append(reward)
                self.results[solver].actions.append(selected_arm)

                # @TODO:Migrate to solver class
                if isinstance(solver, ThomsonSamplingSolver):
                    solver.update_state(selected_arm, reward)

            self.results[solver].cummulatives = self.bandit.get_cumulative_by_arms()
            self.results[solver].usage_fractions = self.bandit.calculate_arm_fractions()
            self.bandit.reset()

    def get_results(self, solver: Solver = None) -> List[float]:
        '''Returns the results for the specified solver.'''
        if solver:
            return self.results.get(solver)
        return self.results

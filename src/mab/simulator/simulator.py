"""Module for running simulations."""
from typing import List
from mab.domain.bandit import Bandit
from mab.domain.solver import Solver

class SimulationResults:
    """Helper class to store the results of the simulation."""
    def __init__(self):
        self.rewards = []  # The rewards for each iteration
        self.actions = []  # The actions for each iteration
        self.usage_fractions = {}  # The fraction of times each arm was selected


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
                reward = self.bandit.pull_arm(selected_arm)
                solver.update_solver_history(selected_arm, reward)
                self.results[solver].rewards.append(reward)
                self.results[solver].actions.append(selected_arm)

            self.results[solver].usage_fractions = self.bandit.calculate_arm_fractions()
            self.bandit.reset()

    def get_results(self, solver: Solver = None) -> List[float]:
        '''Returns the results for the specified solver.'''
        if solver:
            return self.results[solver]
        return self.results

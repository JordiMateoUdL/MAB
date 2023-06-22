from typing import List

from mab.domain.bandit import Bandit
from mab.domain.solver import Solver

class Simulator:
    '''Class representing a simulator for a multi-armed bandit problem.'''
    def __init__(self, bandit: Bandit, solvers: List[Solver]) -> None:
        self.bandit = bandit
        self.solvers = solvers
        self.results = {solver: [] for solver in solvers}

    def run(self, num_iterations: int) -> None:
        '''Runs the simulation for the specified number of iterations.'''
        for solver in self.solvers:
            for _ in range(num_iterations):
                selected_arm = solver.select_arm()
                reward = self.bandit.pull_arm(selected_arm)
                solver.update_solver_history(selected_arm, reward)
                self.results[solver].append(reward)

    def get_solver_results(self, solver: Solver) -> List[float]:
        '''Returns the results for the specified solver.'''
        return self.results[solver]

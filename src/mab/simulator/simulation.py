"""Simulation"""

from mab.domain.bandit import Bandit
from mab.simulator.simulation_results import SimulationResults
from mab.solvers.solvers import BanditAction, BanditSolver


class SimulationController:
    """
    Class to control the execution of multi-armed bandit simulations and store the results.
    """

    def __init__(self) -> None:
        self.simulation_results: SimulationResults = SimulationResults()

    def run_simulation(self, num_time_steps: int, solver: BanditSolver) -> None:
        """
        Run a multi-armed bandit simulation for a given number of time steps.

        Args:
            num_time_steps: The number of time steps to run the simulation.
            solver: An instance of the BanditSolver class used to solve the bandit problem.
        """
        bandit = solver.get_bandit()
        bandit.reset()

        for time_step in range(num_time_steps):
            arm, reward = self.run_time_step(solver)

            cumulative_reward, average_reward = self._get_time_step_metrics(
                bandit, time_step, reward)
        

            self.update_simulation_results(
                cumulative_reward,
                average_reward,
                bandit.get_arm_index(arm),
                arm.get_pull_counts()
            )

    def run_time_step(self, solver: BanditSolver) -> None:
        """
        Run a single time step of the multi-armed bandit simulation.

        Args:
            solver: An instance of the BanditSolver class used to solve the bandit problem.
            bandit: The Bandit object representing the multi-armed bandit.
        """
        arm = solver.select_arm()
        reward = arm.pull()
        solver.update_bandit(arm, reward)
        
        if solver.get_action() == BanditAction.EXPLORATION:
            self.simulation_results.update_exploration_steps()
        else:
            self.simulation_results.update_exploitation_steps()
        
        return arm, reward

    def _get_time_step_metrics(self, bandit: Bandit, time_step: int, reward: float) -> None:
        """
        Get the metrics for the current time step.

        Args:
            bandit: The Bandit object representing the multi-armed bandit.
            time_step: The current time step.
            reward: The reward obtained from pulling the arm at time_step.
        """
        cumulative_reward = bandit.get_cumulative_reward()
        #cumulative_regret = bandit.get_cumulative_regret()
        #instantaneous_regret = bandit.get_regret(reward)
        average_reward = cumulative_reward / (time_step + 1)
        return cumulative_reward, average_reward

    def update_simulation_results(
        self,
        #cumulative_regret: float,
        #instantaneous_regret: float,
        cumulative_reward: float,
        average_reward: float,
        arm_index: int,
        num_pulls: int,
    ) -> None:
        """
        Update the simulation results with the metrics of the current time step.

        Args:
            cumulative_regret: The cumulative regret of the bandit.
            instantaneous_regret: The instantaneous regret of the current time step.
            cumulative_reward: The cumulative reward of the bandit.
            average_reward: The average reward of the bandit.
            arm_index: The index of the selected arm.
            num_pulls: The number of pulls for the selected arm.
        """
        #self.simulation_results.update_regret(
        #    cumulative_regret, instantaneous_regret)
        self.simulation_results.update_reward(
            cumulative_reward, average_reward)
        self.simulation_results.update_arm_pulls(arm_index, num_pulls)

    def get_results(self) -> SimulationResults:
        """
        Get the results of the simulation.

        Returns:
            The SimulationResults object containing the metrics and statistics.
        """
        return self.simulation_results

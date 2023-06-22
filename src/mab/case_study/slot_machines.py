"""Module for the Bernoulli Slot Machines case study."""
from mab.case_study.bernoulli_arm import BernoulliArm
from mab.domain.bandit import Bandit
from mab.simulator.plotter import PlotConfig, Plotter
from mab.simulator.simulator import Simulator
from mab.solvers.epsilon_greedy import EpsilonGreedySolver
from mab.solvers.thomson_sampling import ThomsonSamplingSolver
from mab.solvers.ucb import UCB1Solver

# @TODO: Add a parent abstract case study class
class BernoulliSlotMachines():
    """Bernoulli Slot Machines - Case Study"""
    def __init__(self):
        self.arms = [
            BernoulliArm(0.0),
            BernoulliArm(0.1),
            BernoulliArm(0.2),
            BernoulliArm(0.3),
            BernoulliArm(0.4),
            BernoulliArm(0.5),
            BernoulliArm(0.6),
            BernoulliArm(0.7),
            BernoulliArm(0.8),
            BernoulliArm(0.9)
        ]
        self.bandit = Bandit(self.arms)

        self.epsilon_greedy_solver = EpsilonGreedySolver(
            self.bandit, epsilon=0.01)
        self.ucb1_solver = UCB1Solver(self.bandit, exploration_parameter=1.0)
        self.thomson_sampling_solver = ThomsonSamplingSolver(self.bandit)
        self.solvers = [self.epsilon_greedy_solver,
                        self.ucb1_solver, self.thomson_sampling_solver]

        self.simulator = Simulator(self.bandit, self.solvers)

    def run_simulation(self, iterations):
        """run the simulation for the specified number of iterations."""
        self.simulator.run(iterations)

    def report_results(self):
        """report the results of the simulation."""
        solvers_names = [str(solver) for solver in self.solvers]
        for solver in self.solvers:
            print(f'Solver {solver}')
            results = self.simulator.get_results(solver)
            print(" +% Selection Fraction:")
            for arm in self.bandit.get_arms():
                print(
                    f' ++ Arm {arm}: {results.usage_fractions[arm] * 100:.2f}%')

        benchmark_results = self.simulator.get_results()

        arm_fractions = {}
        rewards = {}
        regrets = {}
        for solver in self.solvers:
            arm_fractions[str(
                solver)] = benchmark_results[solver].usage_fractions.values()
            rewards[str(solver)] = benchmark_results[solver].rewards

            # Calculate the cumulative regret
            regret_history = []
            for estimated_prob in rewards[str(solver)]:
                true_prob = max(arm.success_probability for arm in self.arms)
                regret_history.append(true_prob - estimated_prob)

            regrets[str(solver)] = regret_history

        # Plot the results
        Plotter.plot_arm_selection_fractions(arm_fractions, solvers_names)
        Plotter.show_plot()

        Plotter.plot_cumulative(rewards)
        Plotter.show_plot()

        Plotter.plot_cumulative(
            regrets,
            PlotConfig(x_label="#Iterations",
                       y_label="Cumulative Regret",
                       title="Comparison of cumulative regret obtained by each solver"))
        Plotter.show_plot()

""" Module for simulating slot machines case study """

from mab.case_study.bernoulli_arm import BernoulliArm
from mab.domain.bandit import Bandit
from mab.simulator.plotter import PlotConfig, Plotter
from mab.simulator.simulator import Simulator
from mab.solvers.epsilon_greedy import EpsilonGreedySolver
from mab.solvers.ucb import UCB1Solver


def bernoulli_slot_machines():
    """Bernoulli slot machines case study."""
    # Configuration of the Bernoulli bandit based on the slot machines case study
    arms = [
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

    # Bandit definion for holding the Bernoulli arms
    bandit = Bandit(arms)

    # Solver configuration
    epsilon_greedy_solver = EpsilonGreedySolver(bandit, epsilon=0.01)
    ucb1_solver = UCB1Solver(bandit, exploration_parameter=1.0)
    solvers = [epsilon_greedy_solver, ucb1_solver]

    # Create the simulator instance
    simulator = Simulator(bandit, solvers)

    # Run the simulator for 1000 iterations
    iterations = 5000
    simulator.run(iterations)

    # Report results
    solvers_names = [str(solver) for solver in solvers]
    for solver in solvers:
        print(f'Solver {solver}')
        results = simulator.get_results(solver)
        print(" +% Selection Fraction:")
        for arm in bandit.get_arms():
            print(f' ++ Arm {arm}: {results.usage_fractions[arm] * 100:.2f}%')

    benchmark_results = simulator.get_results()
    

    # Obtain the fraction of times each arm was selected for each solver
    # @TODO: Migrate to Reporter class
    arm_fractions = {}
    rewards = {}
    regrets = {}
    for solver in solvers:
        arm_fractions[str(
            solver)] = benchmark_results[solver].usage_fractions.values()
        rewards[str(
            solver)] = benchmark_results[solver].rewards
        
        # Calculate the cumulative regret
        regret_history = []
        for estimated_prob in rewards[str(solver)]:
            true_prob = max([arm.success_probability for arm in arms])
            regret_history.append(true_prob - estimated_prob)

        regrets[str(solver)] = regret_history
    
    # Plot the results
    # Plot the fraction of times each arm was selected for each solver
    Plotter.plot_arm_selection_fractions(arm_fractions, solvers_names)
    Plotter.show_plot()
    
    Plotter.plot_cumulative(rewards)
    Plotter.show_plot()
    
    Plotter.plot_cumulative(regrets, PlotConfig(x_label="#Iterations", 
                                                y_label="Cumulative Regret", 
                                                title="Comparison of cumulative regret obtained by each solver"))
    Plotter.show_plot()


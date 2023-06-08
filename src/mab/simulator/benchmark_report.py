import matplotlib.pyplot as plt

from mab.simulator.simulation_report import SimulationReporter

class BenchmarkReporter:
    """
    Class to generate plots and reports for benchmarking multiple algorithms.
    """

    def __init__(self):
        self.results = []

    def add_simulation_results(self, simulation_results, algorithm_name):
        """
        Add the simulation results along with the algorithm name to the benchmark reporter.

        Args:
            simulation_results: The SimulationResults object containing the metrics and statistics.
            algorithm_name: The name of the algorithm used in the simulation.
        """
        self.results.append((simulation_results, algorithm_name))

    def plot_cumulative_reward(self):
        plt.figure()
    
        for simulation_results, algorithm_name in self.results:
            cumulative_rewards = simulation_results.get_cumulative_rewards()
            time_steps = range(1, len(cumulative_rewards) + 1)
            plt.plot(time_steps,cumulative_rewards, label=algorithm_name)
            
        plt.xlabel('Time Step')
        plt.ylabel('Cumulative Reward')
        plt.title('Cumulative Reward over Time')
        plt.legend()
        plt.show()


    def generate_report(self):
        report = "----------------------------------------\n"
        report += "Benchmark Results:\n"
        report += "----------------------------------------\n"
        for simulation_results, algorithm_name in self.results:
            report += f"Algorithm: {algorithm_name}\n"
            report += SimulationReporter(simulation_results).generate_report()
            report += "----------------------------------------\n"
        return report
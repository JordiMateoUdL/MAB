import matplotlib.pyplot as plt

from mab.simulation import SimulationResults

class SimulationReporter:
    """
    Class to generate plots and reports based on the simulation results.
    """

    def __init__(self, simulation_results: SimulationResults) -> None:
        self.simulation_results = simulation_results
        
    def generate_reward_plot(self):
        """
        Generate a plot of the cumulative reward over time.
        """
        cumulative_rewards = self.simulation_results.get_cumulative_rewards()
        time_steps = range(1, len(cumulative_rewards) + 1)

        plt.plot(time_steps, cumulative_rewards)
        plt.xlabel("Time Step")
        plt.ylabel("Cumulative Reward")
        plt.title("Cumulative Reward over Time")
        plt.show()

   

    def generate_report(self) -> str:
        """
        Generate a report summarizing the simulation results.
        """
        cumulative_reward = self.simulation_results.get_cumulative_reward()
        average_reward = self.simulation_results.get_average_reward()
        #cumulative_regret = self.simulation_results.get_cumulative_regret()
        exploration_steps = self.simulation_results.get_exploration_steps()
        exploitation_steps = self.simulation_results.get_exploitation_steps()
        arm_pulls = self.simulation_results.get_arm_pulls()

        report = f"Cumulative Reward: {cumulative_reward}\n" \
                 f"Average Reward: {average_reward}\n" \
                 f"Exploration Steps: {exploration_steps}\n" \
                 f"Exploitation Steps: {exploitation_steps}\n" \
                 f"Arm Pulls: {arm_pulls}\n"

        return report
"""Module for helper functions related to plotting the bandit domain problems."""

from typing import Dict, List
import matplotlib.pyplot as plt

class PlotConfig:
    """Helper class to store the plot configuration."""
    def __init__(
        self,
        x_label: str = "Arms",
        y_label: str = "Frac. # Selection",
        x_tick_label: str = "Arm",
        title: str = "Arm Selection Fractions by Solver",
    ):
        self.x_label = x_label
        self.y_label = y_label
        self.x_tick_label = x_tick_label
        self.title = title

class Plotter:

    """ A class used to plot the results of the simulation."""

    def plot_arm_selection_fractions(self,
        arm_fractions: Dict[str, List[float]],
        solver_names: List[str],
        config: PlotConfig = PlotConfig(),
    ) -> plt.Figure:
        """
        Plots the fraction of times each arm was selected during the simulation.

        Args:
            arm_fractions (Dict): A dictionary containing the fraction of times
                each arm was selected.
            solver_names (List[str]): A list containing the names of the solvers.
            config (PlotConfig): An object containing the plot configuration.

        Returns:
            A matplotlib figure with the plot.
        """
        num_arms = len(arm_fractions[solver_names[0]])
        num_solvers = len(solver_names)

        # Plot configuration
        bar_width = 1 / (num_solvers + 1)
        index = range(num_arms)

        # Plot creation
        fig, ax = plt.subplots()
        for i in range(num_solvers):
            ax.bar(
                [x + i * bar_width for x in index],
                arm_fractions[solver_names[i]],
                bar_width,
                label=solver_names[i],
            )

        # Axes and Legend configuration
        ax.set_xlabel(config.x_label)
        ax.set_ylabel(config.y_label)
        ax.set_title(config.title)
        ax.set_xticks(index)
        ax.set_xticklabels([f"{config.x_tick_label} {i+1}" for i in index])
        ax.legend()

        return fig

    def save_plot(self, figure: plt.Figure, filepath: str) -> None:
        """
        Method to export the plot to a file
        Args:
            figure (plt.Figure): The figure to be exported.
            filepath (str): The path to the file.
        """
        figure.savefig(filepath)

    def show_plot(self) -> None:
        """
        Method to show the plot.
        """
        plt.show()

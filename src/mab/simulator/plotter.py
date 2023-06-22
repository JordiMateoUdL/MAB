"""Module for helper functions related to plotting the bandit domain problems."""

from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np


class PlotConfig:
    """Helper class to store the plot configuration."""

    def __init__(
        self,
        x_label: str = "",
        y_label: str = "",
        x_tick_label: str = "",
        title: str = "",
    ):
        self.x_label = x_label
        self.y_label = y_label
        self.x_tick_label = x_tick_label
        self.title = title

    def set_labels(self, x_label: str = "", y_label: str = ""):
        """Sets the labels for the x and y axis."""
        self.x_label = x_label
        self.y_label = y_label

    def set_thick_label(self, x_tick_label: str = ""):
        """Sets the label for the x axis ticks."""
        self.x_tick_label = x_tick_label

    def set_title(self, title: str = ""):
        """Sets the title of the plot."""
        self.title = title


class Plotter:

    """ A class used to plot the results of the simulation."""

    @staticmethod
    def plot_cumulative(
            cummulative: Dict[str, List[float]],
            config: PlotConfig = PlotConfig(
                x_label="# Iterations",
                y_label="Cummulative reward",
                title="Comparative of cumulative rewards obtained by each solver",
            )
    ) -> plt.Figure:
        """Plot the cumulative rewards or regrets obtained by each solver.
        Args:
            cummulative (Dict): A dictionary containing the cumulative rewards/regrets
                obtained by each solver.
            config (PlotConfig): An object containing the plot configuration.
        Returns:
            A matplotlib figure with the plot.
        """

        figure, axes = plt.subplots()

        for solver_name, history_rewards in cummulative.items():
            data = np.cumsum(history_rewards)
            axes.plot(range(len(history_rewards)), data, label=solver_name)

        axes.set_xlabel(config.x_label)
        axes.set_ylabel(config.y_label)
        axes.set_title(config.title)
        axes.legend()

        return figure

    @staticmethod
    def plot_arm_selection_fractions(
        arm_fractions: Dict[str, List[float]],
        solver_names: List[str],
        config: PlotConfig = PlotConfig(
            x_label="Arms",
            y_label="Frac. # Selection",
            x_tick_label="Arm",
            title="Arm Selection Fractions by Solver",
        )
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
        fig, axes = plt.subplots()
        for i in range(num_solvers):
            axes.bar(
                [x + i * bar_width for x in index],
                arm_fractions[solver_names[i]],
                bar_width,
                label=solver_names[i],
            )

        # Axes and Legend configuration
        axes.set_xlabel(config.x_label)
        axes.set_ylabel(config.y_label)
        axes.set_title(config.title)
        axes.set_xticks(index)
        axes.set_xticklabels([f"{config.x_tick_label} {i+1}" for i in index])
        axes.legend()

        return fig

    @staticmethod
    def save_plot(figure: plt.Figure, filepath: str) -> None:
        """
        Method to export the plot to a file
        Args:
            figure (plt.Figure): The figure to be exported.
            filepath (str): The path to the file.
        """
        figure.savefig(filepath)

    @staticmethod
    def show_plot() -> None:
        """
        Method to show the plot.
        """
        plt.show()

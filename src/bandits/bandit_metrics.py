import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

class BanditMetrics:
    def __init__(self, bandit):
        self.bandit = bandit

    def average_reward(self):
        return np.mean(self.bandit.rewards)

    def cumulative_reward(self):
        return np.sum(self.bandit.rewards)

    def optimal_arm_percentage(self):
        best_arm = self.bandit.best_arm
        return self.bandit.strategy.arms[best_arm].reward_counts / float(len(self.bandit.rewards))

    def regret(self):
        optimal_reward = np.max([arm.true_reward for arm in self.bandit.strategy.arms])
        return np.cumsum(optimal_reward - self.bandit.rewards)

    def plot_cumulative_rewards(self):
        rounds = np.arange(1, len(self.bandit.rewards) + 1)
        cumulative_rewards = np.cumsum(self.bandit.rewards)

        plt.plot(rounds, cumulative_rewards)
        plt.xlabel("Round")
        plt.ylabel("Cumulative Reward")
        plt.title("Cumulative Reward over Time")
        plt.show()

    def plot_histogram_arm_selections(self):
        arm_counts = [arm.reward_counts for arm in self.bandit.strategy.arms]
        arm_labels = [f"Arm {i+1}" for i in range(len(self.bandit.strategy.arms))]

        plt.bar(arm_labels, arm_counts)
        plt.xlabel("Arm")
        plt.ylabel("Number of Selections")
        plt.title("Histogram of Arm Selections")
        plt.show()



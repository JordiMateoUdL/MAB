
import random
from enum import Enum
from abc import ABC, abstractmethod
from mab.domain.arm import Arm

from mab.domain.bandit import Bandit

class BanditAction(Enum):
    EXPLORATION = "Exploration"
    EXPLOITATION = "Exploitation"
    

class BanditSolver(ABC):
    def __init__(self, bandit: Bandit):
        self._bandit = bandit
        self._action = None

    @abstractmethod
    def select_arm(self) -> Arm:
        pass
    
    def get_bandit(self) -> Bandit:
        return self._bandit
    
    def update_bandit(self, arm: Arm, reward: float) -> None:
        self._bandit.update(arm, reward)
        
    def get_action(self) -> BanditAction:
        return self._action
        
class EpsilonGreedy(BanditSolver):
    def __init__(self, bandit, epsilon):
        super().__init__(bandit)
        self._epsilon = epsilon

    def select_arm(self) -> Arm:

        arms = self._bandit.get_arms()

        if random.random() < self._epsilon:
            # Explore: Select a random arm
            self._action = BanditAction.EXPLORATION
            return random.choice(arms)

        # Exploit: Select the arm with the highest average reward
        self._action = BanditAction.EXPLOITATION
        return max(arms, key=lambda arm: arm.get_cumulative_reward())
    
    def __str__(self) -> str:
        return f" EpsilonGreedy(epsilon={self._epsilon})"
    

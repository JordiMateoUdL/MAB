from abc import ABC, abstractmethod
from typing import Any, List, Union
import numpy as np
from domain.arm import Arm

class Strategy(ABC):
    
    """
    Abstract base class representing a strategy for selecting an arm in a multi-armed bandit problem.
    """
    
    def __init__(self, arms:List[Arm]):
        self.arms:List[Arm] = arms
        self.num_arms:int = len(arms)
        self.best_reward:Union(int,float) = np.max([arm.true_reward for arm in arms])
        self.exploration_steps:int = 0
        self.exploitation_steps:int = 0
        pass
    
    @abstractmethod
    def select_arm(self) -> int:
        """
        Abstract method for selecting an arm based on the strategy.

        Returns:
            The index of the selected arm.
        """
        
        raise NotImplementedError("select_arm method must be implemented...")
        
    
class EpsilonGreedy(Strategy):
    
    """
    Epsilon-Greedy strategy for selecting an arm
    """
    
    def __init__(self, arms:List[Arm], epsilon:float=0.1):
        """
        Initializes the EpsilonGreedy strategy.

        Args:
            arms (List[Arm]): The list of arms available.
            epsilon (float): The exploration rate (0 to 1) for balancing exploration and exploitation.
        """
        super().__init__(arms=arms)
        self.epsilon:float = epsilon
    
    def select_arm(self) -> int:
        """
        Selects an arm based on the Epsilon-Greedy strategy.

        Returns:
            The index of the selected arm.
        """
        if np.random.random() < self.epsilon:
            self.exploration_steps += 1
            return np.random.randint(0, self.num_arms)
        else:
            self.exploitation_steps += 1
            return np.argmax([arm.cumulative_reward for arm in self.arms])
    
    def __str__(self) -> str:
        """
        Returns:
            A string representing the EpsilonGreedy strategy.
        """
        return f'EpsilonGreedy(e={self.epsilon})'
    

class UCB1(Strategy):
    
    def __init__(self, arms:List[Arm], c:Union[int,float]=1.0):
        """
        Initializes the UCB1 strategy.

        Args:
            arms (List[Arm]): The list of arms available.
            c (Union[int,float]): The exploration parameter
        """
        super().__init__(arms=arms)
        self.t:int = 0 
        self.c:Union(int,float) = c
        
    def __str__(self) -> str:
        return f'UCB1(c={self.c})'
    
    def select_arm(self) -> int:
        
        seleceted = np.argmax([arm.cumulative_reward + ( (self.c * np.sqrt(np.log(self.t)) / (arm.pull_counts + 1e-8)) )
                               for arm in self.arms])     
        self.t += 1
        return seleceted
    
class PureExploration(Strategy):
    
    """
    Pure exploration strategy for selecting an arm. Always selects a random arm.
    """
    
    def __init__(self,arms:List[Arm]):
        """
        Initializes the PureExploration strategy.

        Args:
            arms (List[Arm]): The list of arms available in the bandit problem.
        """
        super().__init__(arms=arms)
        
    def select_arm(self):
        """
        Selects an arm based on the PureExploration strategy.

        Returns:
            The index of the selected arm.
        """
        self.exploration_steps += 1
        return np.random.randint(0, self.num_arms)
    
    def __str__(self) -> str:
        """
        Returns:
            A string representing the PureExploration strategy.
        """
        return "PureExploration"

class PureExploitation(Strategy):
    """
    Pure exploitation strategy for selecting an arm. Always selects the arm with the highest cumulative reward.
    """
    
    def __init__(self,arms:List[Arm]):
        """
        Initializes the PureExploration strategy.

        Args:
            arms (List[Arm]): The list of arms available.
        """
        super().__init__(arms=arms)
        
    def select_arm(self):
        """
        Selects an arm based on the PureExploration strategy.
        Returns:
            The index of the selected arm.
        """
        self.exploitation_steps += 1
        return np.argmax([arm.cumulative_reward for arm in self.arms])
    
    def __str__(self) -> str:
        """
        Returns:
            A string representing the PureExploration strategy.
        """
        return "PureExploitation"
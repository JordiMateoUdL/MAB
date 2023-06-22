"""Test module for the Solver class."""""
import unittest
from unittest.mock import MagicMock
from mab.domain.arm import Arm
from mab.domain.bandit import Bandit
from mab.domain.solver import Solver, SolverAction

class SolverMock(Solver):
    """Solver mock implementation for testing purposes."""
    def select_arm(self) -> int:
        return 0

class SolverTestCase(unittest.TestCase):
    """Test cases for the Solver class."""
    def setUp(self):
        self.bandit = MagicMock(spec=Bandit)
        self.solver = SolverMock(self.bandit)

    def test_update_solver_history(self):
        """Test the update_solver_history method."""
        arm = MagicMock(spec=Arm)
        self.solver.update_solver_history(arm, SolverAction.EXPLORE)
        history = self.solver.get_action_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0], (arm, SolverAction.EXPLORE))

    def test_get_action_history(self):
        """Test the get_action_history method."""
        arm1 = MagicMock(spec=Arm)
        arm2 = MagicMock(spec=Arm)
        self.solver.update_solver_history(arm1, SolverAction.EXPLORE)
        self.solver.update_solver_history(arm2, SolverAction.EXPLOIT)
        history = self.solver.get_action_history()
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0], (arm1, SolverAction.EXPLORE))
        self.assertEqual(history[1], (arm2, SolverAction.EXPLOIT))

if __name__ == '__main__':
    unittest.main()

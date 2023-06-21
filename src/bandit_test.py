"""
Test cases for the Bandit class.
"""
import unittest
from arm_test import MOCK_REWARD, PULLS, MockArm
from mab.domain.bandit import Bandit


class BanditTestCase(unittest.TestCase):

    '''Test cases for the Bandit class.'''

    def setUp(self):
        self.bandit = Bandit()
        self.arm1 = MockArm()
        self.arm2 = MockArm()

        # Add arms to the bandit
        self.bandit.add_arm(self.arm1)
        self.bandit.add_arm(self.arm2)

    def tearDown(self):
        self.bandit = None
        self.arm1 = None
        self.arm2 = None

    def test_get_arms(self):
        '''Test the get_arms method.'''
        self.assertEqual(self.bandit.get_arms(), [self.arm1, self.arm2])

    def test_get_arms_number(self):
        '''Test the get_arms_number method.'''
        self.assertEqual(self.bandit.get_arms_number(), 2)

    def test_get_arm(self):
        '''Test the get_arm method.'''
        self.assertEqual(self.bandit.get_arm(0), self.arm1)
        self.assertEqual(self.bandit.get_arm(1), self.arm2)

    def test_get_arm_index(self):
        '''Test the get_arm_index method.'''
        self.assertEqual(self.bandit.get_arm_index(self.arm1), 0)
        self.assertEqual(self.bandit.get_arm_index(self.arm2), 1)

    def test_add_arm(self):
        '''Test the add_arm method.'''

        # Create a new arm
        new_arm = MockArm()

        # Add the new arm to the bandit
        self.bandit.add_arm(new_arm)

        # Assert that the new arm is added to the bandit
        self.assertIn(new_arm, self.bandit.get_arms())

    def test_remove_arm(self):
        '''Test the remove_arm method.'''

        # Remove one of the arms from the bandit
        self.bandit.remove_arm(self.arm1)

        # Assert that the removed arm is not in the bandit
        self.assertNotIn(self.arm1, self.bandit.get_arms())


    def test_get_cumulative_reward(self):
        '''Test the get_cumulative_reward method.'''

        # Pull both arms multiple times
        for _ in range(PULLS):
            self.bandit.pull_arm(self.arm1)
            self.bandit.pull_arm(self.arm2)

        # Calculate the expected cumulative reward
        expected_cumulative_reward = MOCK_REWARD * PULLS * 2

        # Assert that the returned cumulative reward matches the expected value
        self.assertEqual(
            self.bandit.calculate_cumulative_reward(), expected_cumulative_reward)

    def test_reset(self):
        '''Test the reset method.'''

        # Pull both arms multiple times
        for _ in range(PULLS):
            self.bandit.pull_arm(self.arm1)
            self.bandit.pull_arm(self.arm2)

        # Reset the bandit
        self.bandit.reset()

        # Assert that the arms' pull counts have been reset to zero
        self.assertEqual(self.arm1.get_pull_counts(), 0)
        self.assertEqual(self.arm2.get_pull_counts(), 0)

        # Assert that the bandit's cumulative reward has been reset to zero
        self.assertEqual(self.bandit.calculate_cumulative_reward(), 0)

    def test_clone(self):
        '''Test the clone method.'''

        # Clone the bandit
        cloned_bandit = self.bandit.__clone__()

        # Assert that the cloned bandit is not the same as the original bandit
        assert cloned_bandit is not self.bandit

        assert self.bandit.get_arms_number() == cloned_bandit.get_arms_number()
        assert self.bandit.calculate_cumulative_reward(
        ) == cloned_bandit.calculate_cumulative_reward()
        assert self.bandit.get_arms() is not cloned_bandit.get_arms()
        assert self.bandit.get_arm(0) is not cloned_bandit.get_arm(0)
        assert self.bandit.get_arm(1) is not cloned_bandit.get_arm(1)

    def test_pull_arm(self):
        '''Test the pull_arm method.'''

        # Pull an arm from the bandit
        reward = self.bandit.pull_arm(self.arm1)

        # Assert that the returned reward matches the mock reward value
        self.assertEqual(reward, MOCK_REWARD)

        # Assert that the arm's pull counts have been updated
        self.assertEqual(self.arm1.get_pull_counts(), 1)
        
        # Assert that the arm cumulative reward has been updated
        self.assertEqual(self.arm1.get_cumulative_reward(), MOCK_REWARD)

        # Test pulling an arm that is not in the bandit
        invalid_arm = MockArm()
        with self.assertRaises(ValueError):
            self.bandit.pull_arm(invalid_arm)

if __name__ == '__main__':
    unittest.main()

'''
Test cases for the Arm class
'''
import unittest
from mab.domain.arm import Arm

# Constants for the tests
MOCK_REWARD: int = 5
PULLS: int = 3


class MockArm(Arm):
    ''' Mock implementation of the Arm class for testing purposes.'''

    def pull(self):
        '''pull method that returns a mock reward value of 5'''
        super().pull()
        return MOCK_REWARD

    def update_cumulative_reward(self, reward):
        '''updates the cumulative reward adding the reward value.'''
        self._cumulative_reward += reward


class ArmTestCase(unittest.TestCase):

    '''Test cases for the Arm class.'''

    def setUp(self):
        self.arm = MockArm()

    def tearDown(self):
        self.arm = None

    def test_pull(self):
        '''Test the pull method.'''

        # Call the pull method
        reward = self.arm.pull()

        # Assert that the returned reward matches the mock reward value
        self.assertEqual(reward, MOCK_REWARD)

        # Assert that pull_counts has been updated
        self.assertEqual(self.arm.get_pull_counts(), 1)

    def test_update_cumulative_reward(self):
        '''Test the update_cumulative_reward method.'''

        # Update the reward with a mock value
        self.arm.update_cumulative_reward(MOCK_REWARD)
        self.arm.update_cumulative_reward(MOCK_REWARD)

        # Assert that the cumulative reward has been updated correctly
        self.assertEqual(self.arm.get_cumulative_reward(), 2*MOCK_REWARD)

        # Assert that the pull counts remain unchanged
        self.assertEqual(self.arm.get_pull_counts(), 0)

    def test_reset(self):
        '''Test the reset method.'''

        # Set up initial values

        # Do operations with the arm
        self.arm.update_cumulative_reward(MOCK_REWARD)
        for _ in range(PULLS):
            self.arm.pull()

        # Reset the arm
        self.arm.reset()

        # Assert that the cumulative reward and pull counts have been reset to zero
        self.assertEqual(self.arm.get_cumulative_reward(), 0)
        self.assertEqual(self.arm.get_pull_counts(), 0)

    def test_get_pull_counts(self):
        '''Test the get_pull_counts method.'''

        # Simulate pulling the arm multiple times
        for _ in range(PULLS):
            self.arm.pull()

        # Assert that the returned pull counts match the initial pull counts
        self.assertEqual(self.arm.get_pull_counts(), PULLS)

    def test_get_cumulative_reward(self):
        '''Test the get_cumulative_reward method.'''

        # Update the reward with a mock value
        self.arm.update_cumulative_reward(MOCK_REWARD)

        # Assert that the returned cumulative reward matches the initial reward
        self.assertEqual(self.arm.get_cumulative_reward(), MOCK_REWARD)

    def test_clone(self):
        '''Test the __clone__ method.'''

        # Do operation to create a arm's state
        self.arm.pull()
        self.arm.update_cumulative_reward(MOCK_REWARD)

        # Clone the arm
        cloned_arm = self.arm.__clone__()

        # Verify that the cloned arm is a separate instance
        self.assertIsNot(self.arm, cloned_arm)

        # Verify that the cloned arm has the same pull counts and cumulative reward
        self.assertEqual(self.arm.get_pull_counts(),
                         cloned_arm.get_pull_counts())
        self.assertEqual(self.arm.get_cumulative_reward(),
                         cloned_arm.get_cumulative_reward())


if __name__ == '__main__':
    unittest.main()

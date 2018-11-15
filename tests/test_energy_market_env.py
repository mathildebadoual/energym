import numpy as np
import unittest
import gym
import energym


class TestEnergyMarketEnv(unittest.TestCase):
    def setUp(self):
        self.env = gym.make('energy_market-v0')

    def test_as_gym_env(self):
        ob_space = self.env.observation_space
        act_space = self.env.action_space
        ob = self.env.reset()
        self.assertTrue(ob_space.contains(ob))

        a = np.array([1000, 2])
        observation, reward, done, _info = self.env.step(a)
        self.assertTrue(ob_space.contains(observation))
        self.assertTrue(np.isscalar(reward))
        self.assertTrue(isinstance(done, bool))

    def test_step(self):
        pass
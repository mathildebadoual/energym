import numpy as np
import unittest
import gym
import energym


class TestEnergyMarketEnv(unittest.TestCase):
    def setUp(self):
        self.env = gym.make('energy_market_battery-v0')

    def test_as_gym_env(self):
        ob_space = self.env.observation_space
        ob = self.env.reset()
        self.assertTrue(ob_space.contains(ob))

        a = self.env.action_space.sample()
        observation, reward, done, _info = self.env.step(a)
        self.assertTrue(ob_space.contains(observation))
        self.assertTrue(np.isscalar(reward))
        self.assertTrue(isinstance(done, bool))

    def test_step(self):
        self.env.reset()
        for i in range(100):
            action = self.env.action_space.sample()
            ob_next, reward, done, _ = self.env.step(action)
        self.assertTrue(reward <= 0)

    def test_discrete_to_continuous_action(self):
        action = self.env.action_space.sample()
        quantity, cost = self.env._discrete_to_continuous_action(action)
        self.assertIsInstance(quantity, float)
        self.assertIsInstance(cost, float)
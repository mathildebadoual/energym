import numpy as np
import unittest
import gym
import energym
import datetime
from contextlib import contextmanager
from energym.envs.grid_scale.utils import OptimizationException


class TestEnergyMarketEnv(unittest.TestCase):
    def setUp(self):
        self.env = gym.make('energy_market-v0')

    def test_as_gym_env(self):
        ob_space = self.env.observation_space
        act_space = self.env.action_space
        ob = self.env.reset()
        self.assertTrue(ob_space.contains(ob))

        a = np.array([369, 31])
        observation, reward, done, _info = self.env.step(a)
        self.assertTrue(ob_space.contains(observation))
        self.assertTrue(np.isscalar(reward))
        self.assertTrue(isinstance(done, bool))

    def test_step(self):
        start_date = datetime.datetime(2017, 8, 2, 5, 00, 00)
        self.env.reset(start_date=start_date)
        action = np.array([-800, 4.8])
        with self.assertNotRaises(OptimizationException):
            self.env.step(action)

    @contextmanager
    def assertNotRaises(self, exc_type):
        try:
            yield None
        except exc_type:
            raise self.failureException('{} raised'.format(exc_type.__name__))
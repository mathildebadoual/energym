import numpy as np
import unittest
import gym
import energym

# This is a code inspired by OpenAI.gym tests code


class TestBatteryEnv(unittest.TestCase):
    def setUp(self):
        self.env = gym.make('battery-v0')

    def test_as_gym_env(self):
        ob_space = self.env.observation_space
        act_space = self.env.action_space
        ob = self.env.reset()
        self.assertTrue(ob_space.contains(ob))

        a = act_space.sample()
        observation, reward, done, _info = self.env.step(a)
        self.assertTrue(ob_space.contains(observation))
        self.assertTrue(np.isscalar(reward))
        self.assertTrue(isinstance(done, bool))

        # Run a longer rollout
        agent = lambda ob: self.env.action_space.sample()
        ob = self.env.reset()
        for _ in range(10):
            self.assertTrue(self.env.observation_space.contains(ob))
            a = agent(ob)
            self.assertTrue(self.env.action_space.contains(a))
            (ob, _reward, done, _info) = self.env.step(a)
            print(ob)
            if done:
                break

    def test_step(self):
        action = 0
        ob = self.env.reset()
        ob_next, reward, done, _ = self.env.step(action)
        self.assertEqual(ob, ob_next)

        action = np.array([2])
        ob = self.env.reset()
        ob_next, reward, done, _ = self.env.step(action)
        self.assertEqual(ob + self.env._efficiency_ratio * action, ob_next)

        action = 10000
        self.env.reset()
        ob_next, reward, done, _ = self.env.step(action)
        self.assertEqual(reward, -10000000)
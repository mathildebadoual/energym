import numpy as np
import unittest
import gym
import energym
from energym.envs import battery_env, energy_market_env

# This is a code inspired by OpenAI.gym tests code

class TestBatteryEnv(unittest.TestCase):
    def setUp(self):
        self.env = gym.make('battery-v0')

    def test_battery_env(self):
        ob_space = self.env.observation_space
        act_space = self.env.action_space
        ob = self.env.reset()
        self.assertTrue(ob_space.contains(ob))

        a = act_space.sample()
        observation, reward, done, _info = self.env.step(a)
        self.assert ob_space.contains(observation), 'Step observation: {!r} not in space'.format(observation)
        self.assert np.isscalar(reward), "{} is not a scalar for {}".format(reward, env)
        self.assert isinstance(done, bool), "Expected {} to be a boolean".format(done)

        self.assert

        env.close()

# Run a longer rollout on some environments
def test_random_rollout():
    for env in [envs.make('CartPole-v0'), envs.make('FrozenLake-v0')]:
        agent = lambda ob: env.action_space.sample()
        ob = env.reset()
        for _ in range(10):
            assert env.observation_space.contains(ob)
            a = agent(ob)
            assert env.action_space.contains(a)
            (ob, _reward, done, _info) = env.step(a)
            if done: break
        env.close()

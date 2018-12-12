import gym
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding

import logging
logger = logging.getLogger(__name__)


class BatteryEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        # limitation parameters
        self._max_soe = 3000  # soe = state of energy in MWh
        self._min_soe = 0
        self._max_power = 1000  # power to charge or discharge in MW
        self._min_power = -1000
        self._efficiency_ratio = 0.99

        # cost when close to limits
        self._cost_lim_power = 100
        self._cost_lim_soe = 100

        # gym variables
        self.observation_space = spaces.Box(low=self._min_soe, high=self._max_soe, shape=(1,), dtype=np.float32)
        # TODO(Mathilde): Add option for a discrete action space (for now it is continuous)
        self.action_space = spaces.Box(low=self._min_power, high=self._max_power, shape=(1,), dtype=np.float32)

        # The state is the soc
        self._state = np.array([0])

        self.reset()

    def step(self, action):
        if not isinstance(action, np.ndarray):
            action = np.array([action])

        reward = self.get_penalty(action)

        energy_to_add =  - self._efficiency_ratio * action
        next_soe = self._state + energy_to_add

        if self.action_space.contains(abs(energy_to_add)) and self.observation_space.contains(next_soe):
            self._state = next_soe

        ob = self._get_obs()

        # TODO(Mathilde): define when it is done for this env
        done = False
        return ob, reward, done, dict()

    def reset(self):
        # TODO(Mathilde): look at the gym.spaces.Box doc to see if we can randomly select a element in the "box"
        self._state = self.observation_space.sample()
        # self._state = np.array([0])
        return self._get_obs()

    def render(self, mode='rgb_array'):
        print('current state:', self._state)

    def _get_obs(self):
        # to assure we are not overwriting on the state
        return np.copy(self._state)

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed = seed
        else:
            np.random.seed = seeding.np_random()

    def get_parameters_battery(self):
        return self._max_soe, self._min_soe, self._max_power, self._min_power, self._efficiency_ratio

    def get_penalty(self, power):
        if not isinstance(power, np.ndarray):
            power = np.array([power])
        energy_to_add = self._efficiency_ratio * power
        next_soe = self._state - energy_to_add
        reward = 0
        cost_lim_power = self._cost_lim_power * (min(self._max_power - energy_to_add[0], 0) + min(energy_to_add[0] - self._min_power, 0))
        cost_lim_soe = self._cost_lim_soe * (min(self._max_soe - next_soe[0], 0) + min(next_soe[0] - self._min_soe, 0))
        # if self.action_space.contains(abs(energy_to_add)) and self.observation_space.contains(next_soe):
        #     reward = -1000
        reward = cost_lim_power + cost_lim_soe
        return reward

    def is_safe(self, action):
        if not isinstance(action, np.ndarray):
            action = np.array([action])
        energy_to_add = -self._efficiency_ratio * action
        next_soe = self._state + energy_to_add
        if self.action_space.contains(abs(energy_to_add)) and self.observation_space.contains(next_soe):
            return True
        return False

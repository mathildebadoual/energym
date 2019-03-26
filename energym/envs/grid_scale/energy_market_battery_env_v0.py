import gym
import numpy as np
import datetime
from gym import error, spaces, utils
from gym.utils import seeding
from energym.envs.grid_scale.utils import OptimizationException, EmptyDataException

import logging
logger = logging.getLogger(__name__)

# TODO(Mathilde): Explain well what is this environment...


class EnergyMarketBatteryEnv(gym.Env):
    def __init__(self):
        # TODO(Mathilde): If different modes, should take only continous envs + delta_time and start_date not defined here ...
        self._energy_market = gym.make('energy_market-v0')
        self._battery = gym.make('battery-v0')
        self._state = np.array([0, 0, 0, 0, 0], dtype=np.float32)
        self._n_discrete_cost = 200
        self._n_discrete_power = 200
        self._n_discrete_actions = self._n_discrete_power * self._n_discrete_cost
        self._min_cost, self._max_cost = 0, 100
        self._min_power, self._max_power = self._battery.action_space.low[0], self._battery.action_space.high[0]

        # gym variables
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(self._battery.observation_space.shape[0] +
                                                   self._energy_market.observation_space.shape[0],),
                                            dtype=np.float32)
        self.action_space = spaces.Discrete(self._n_discrete_actions)

    def step(self, action):
        if isinstance(action, int) or isinstance(action, np.int64):
            action = int(action)
            power, cost = self.discrete_to_continuous_action(action)
        else:
            power, cost = action[0], action[1]
            
        done = False

        # first put the penalty (shielding)
        reward = self._battery.get_penalty(np.array([power]))

        try:
            ob_market, _, done, info_market = self._energy_market.step(np.array([power, cost]))
        except OptimizationException:
            self._state = np.zeros(self.observation_space.shape[0])
            ob = self._get_obs()
            print('optimization exception')
            return ob, reward, done, dict()

        power_cleared = ob_market[0]

        date = self._energy_market._date

        if -2 < power_cleared < 2 and power != 0:
            power_cleared = 0
            reward += -50

        ob_battery, reward_battery, _, _ = self._battery.step(power_cleared)

        # define state and reward
        self._state = np.concatenate((ob_market, ob_battery))
        if reward_battery == 0 and not done:
            reward += min(power_cleared, 0) * info_market['price_cleared']
            reward += max(power_cleared, 0) * cost
        ob = self._get_obs()

        return ob, reward, done, dict({'date': date, 'price_cleared': info_market['price_cleared'], 'ref_price': info_market['ref_price'], 'price_bid': cost})

    def reset(self, start_date=None):
        ob_market = self._energy_market.reset(start_date=start_date)
        ob_battery = self._battery.reset()
        self._state = np.concatenate((ob_market, ob_battery))
        return self._get_obs()

    def set_test(self):
        self._energy_market.set_test()

    def render(self, mode='rgb_array'):
        print('Current state:', self._state)

    def _get_obs(self):
        # to assure we are not overwriting on the state
        return np.copy(self._state)

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed = seed
        else:
            np.random.seed = seeding.np_random()

    @property
    def _power_precision(self):
        return (self._max_power - self._min_power) / self._n_discrete_power

    @property
    def _cost_precision(self):
        return (self._max_cost - self._min_cost) / self._n_discrete_cost

    def discrete_to_continuous_action(self, discrete_action):
        """
        maps the integer discrete_action to the grid (power, cost)
        :param discrete_action: int
        :return: (float, float)
        """
        power = self._min_power + (discrete_action % self._n_discrete_power) * self._power_precision
        cost = self._min_cost + (discrete_action // self._n_discrete_power) * self._cost_precision

        return power, cost

    def is_safe(self, action_index):
        power, cost = self.discrete_to_continuous_action(action_index)
        return self._battery.is_safe(power)

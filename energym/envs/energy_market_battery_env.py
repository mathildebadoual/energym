import gym
import numpy as np
import datetime
from gym import error, spaces, utils
from gym.utils import seeding
from energym.envs.utils import OptimizationException, EmptyDataException

import logging
logger = logging.getLogger(__name__)

# TODO(Mathilde): Explain well what is this environment...


class EnergyMarketBatteryEnv(gym.Env):
    def __init__(self, start_date=datetime.datetime(2017, 7, 3), delta_time=datetime.timedelta(hours=1)):
        # TODO(Mathilde): If different modes, should take only continous envs
        self._energy_market = gym.make('energy_market-v0')
        self._battery = gym.make('battery-v0')
        self._start_date = start_date
        self._state = np.array([0, 0, 0, 0, 0], dtype=np.float32)
        self._delta_time = delta_time
        self._n_discrete_cost = 50
        self._n_discrete_power = 50
        self._n_discrete_actions = self._n_discrete_power * self._n_discrete_cost
        self._min_cost, self._max_cost = 0, 20
        self._min_power, self._max_power = self._battery.action_space.low[0], self._battery.action_space.high[0]

        # gym variables
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(self._battery.observation_space.shape[0] +
                                                   self._energy_market.observation_space.shape[0],),
                                            dtype=np.float32)
        self.action_space = spaces.Discrete(self._n_discrete_actions)

    def step(self, action):
        power, cost = self._discrete_to_continuous_action(action)
        done = False
        reward = 0

        try:
            ob_market, _, _, _ = self._energy_market.step(np.array([power, cost]))
        except OptimizationException:
            self._state = np.array([0, 0, 0], dtype=np.float32)
            ob = self._get_obs()
            print('optimization exception')
            return ob, reward, done, dict()
        except EmptyDataException:
            print("end of data, resetting the environment...")
            self._state = np.array([0, 0, 0], dtype=np.float32)
            ob = self._get_obs()
            done = True
            return ob, reward, done, dict()

        power_cleared = ob_market[0]

        ob_battery, reward_battery, _, _ = self._battery.step(power_cleared)

        # define state and reward
        self._state = np.concatenate((ob_battery, ob_market))
        reward = abs(power_cleared) * cost + reward_battery
        ob = self._get_obs()

        return ob, reward, done, dict()

    def reset(self, start_date=None):
        if start_date is not None:
            self._start_date = start_date
        ob_market = self._energy_market.reset(self._start_date)
        ob_battery = self._battery.reset()
        self._state = np.concatenate((ob_market, ob_battery))
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

    @property
    def _power_precision(self):
        return (self._max_power - self._min_power) / self._n_discrete_power

    @property
    def _cost_precision(self):
        return (self._max_cost - self._min_cost) / self._n_discrete_cost

    def _discrete_to_continuous_action(self, discrete_action):
        """
        maps the integer discrete_action to the grid (power, cost)
        :param discrete_action: int
        :return: (float, float)
        """
        power = self._min_power + (discrete_action % self._n_discrete_power) * self._power_precision
        cost = self._min_cost + (discrete_action // self._n_discrete_power) * self._cost_precision

        return power, cost
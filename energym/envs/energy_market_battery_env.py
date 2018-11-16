import gym
import numpy as np
import datetime
from gym import error, spaces, utils
from gym.utils import seeding

import logging
logger = logging.getLogger(__name__)

# TODO(Mathilde): Explain well what is that environment


class EnergyMarketBatteryEnv(gym.Env):
    def __init__(self, start_date=datetime.datetime(2017, 7, 3), delta_time=datetime.timedelta(hours=1)):
        self._energy_market = gym.make('energy_market-v0')
        self._battery = gym.make('battery-v0')
        self._start_date = start_date
        self._state = np.array([0, 0, 0, 0, 0], dtype=np.float32)
        self._delta_time = delta_time

        # gym variables
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(self._battery.observation_space.shape[0] +
                                                   self._energy_market.observation_space.shape[0],),
                                            dtype=np.float32)
        self.action_space = self._battery.action_space

    def step(self, action):
        quantity, price = action[0], action[1]
        done = False
        reward = 0

        try:
            ob_market, _, _, _ = self._energy_market.step(action)
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

        quantity_cleared, cleared_bool = ob_market[0], ob_market[1]

        ob_battery, reward_battery, _, _ = self._battery.step(quantity_cleared * cleared_bool)

        # define state and reward
        self._state = np.concatenate((ob_battery, ob_market))
        reward = quantity_cleared * price * cleared_bool + reward_battery
        ob = self._get_obs()

        return ob, reward, done, dict()

    def reset(self):
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

class EmptyDataException(Exception):
    def __init__(self):
        super().__init__()


class OptimizationException(Exception):
    def __init__(self):
        super().__init__()
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
    def __init__(self):
        # TODO(Mathilde): If different modes, should take only continous envs + delta_time and start_date not defined here ...
        self._energy_market = gym.make('energy_market-v0')
        self._battery = gym.make('battery-v0')
        self._start_date = self._energy_market.get_start_date()
        self._state = np.array([0, 0, 0, 0, 0], dtype=np.float32)
        self._delta_time = delta_time=datetime.timedelta(hours=1)
        self._date = datetime.timedelta(hours=1)
        self._n_discrete_cost = 200
        self._n_discrete_power = 200
        self._n_discrete_actions = self._n_discrete_power * self._n_discrete_cost
        self._min_cost, self._max_cost = 0, 100
        self._min_power, self._max_power = self._battery.action_space.low[0], self._battery.action_space.high[0]

        # gym variables
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(self._battery.observation_space.shape[0] +
                                                   self._energy_market.observation_space.shape[0] + 1,),
                                            dtype=np.float32)
        self.action_space = spaces.Discrete(self._n_discrete_actions)

    def get_start_date(self):
        return self._start_date

    def step(self, action):
<<<<<<< HEAD
        if isinstance(action, int) or isinstance(action, np.int64):
            action = int(action)
=======
        if type(action) == int or True:
>>>>>>> 5d59957420601e9dbabbf4f5a63a59a2b8d5e8e4
            power, cost = self.discrete_to_continuous_action(action)
        else:
            power, cost = action[0], action[1]
            
        done = False
        reward = 0

        # first put the penalty (shielding)
        reward = self._battery.get_penalty(np.array([power]))

        try:
            ob_market, _, done, info_market = self._energy_market.step(np.array([power, cost]))
        except OptimizationException:
            self._state = np.zeros(self.observation_space.shape[0])
            ob = self._get_obs()
            print('optimization exception')
            self._date += self._delta_time
            return ob, reward, done, dict()

        power_cleared = ob_market[0]

        self._date += self._delta_time

        if -5 < power_cleared < 5:
            power_cleared = 0
            reward = - 100

        ob_battery, reward_battery, _, _ = self._battery.step(power_cleared)

        # define state and reward
        self._state = np.concatenate((ob_market, ob_battery, np.array(info_market['ref_price']).reshape((1,))))
        reward += reward_battery
        if reward_battery >= 0 and not done:
            if power_cleared >= 0:
                reward += power_cleared * cost
            else:
                reward += power_cleared * info_market['price_cleared']
        ob = self._get_obs()

        return ob, reward, done, dict({'date': self._date, 'price_cleared': info_market['price_cleared'], 'ref_price': info_market['ref_price']})

    def reset(self, start_date=None):
        if start_date is not None:
            self._date = start_date
        else:
            self._date = self._start_date
        ob_market = self._energy_market.reset(self._date)
        ob_battery = self._battery.reset()
        self._state = np.concatenate((ob_market, ob_battery, np.array([0]).reshape((1,))))
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

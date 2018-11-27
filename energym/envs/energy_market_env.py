import gym
from gym import spaces
from gym.utils import seeding
from energym.envs.utils import OptimizationException, EmptyDataException

import pandas as pd
import cvxpy as cvx
import numpy as np
import pytz
import datetime
import logging
import os

logger = logging.getLogger(__name__)


class EnergyMarketEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, start_date=datetime.datetime(2017, 7, 3), delta_time=datetime.timedelta(hours=1)):
        self._num_agents = 5
        self._date = start_date
        self._start_date = start_date
        self._delta_time = delta_time
        data_path = os.path.join(os.path.dirname(__file__), 'data')
        self._opt_problem = self.build_opt_problem()
        self._gen_df = pd.read_pickle(data_path + "/gen_caiso.pkl")
        self._dem_df = pd.read_pickle(data_path + "/dem_caiso.pkl")
        self._timezone = pytz.timezone("America/Los_Angeles")
        self._print_optimality = False

        # gym variables
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
        # TODO(Mathilde): Add an option for a discrete action space
        self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)

        # the state is the clearance (bool) and the quantity cleared
        self._state = np.array([0], dtype=np.float32)

        self.reset()

    def build_opt_problem(self):
        # build parameters
        self._p_max = cvx.Parameter(self._num_agents)
        self._p_min = cvx.Parameter(self._num_agents)
        self._cost = cvx.Parameter(self._num_agents)
        self._demand = cvx.Parameter()

        # build variables
        self._p = cvx.Variable(self._num_agents)

        # build constraints
        constraint = [np.ones(self._num_agents).T * self._p == self._demand]
        for i in range(self._num_agents):
            constraint += [self._p[i] <= self._p_max[i]]
            constraint += [self._p_min[i] <= self._p[i]]

        # build the objective
        objective = cvx.Minimize(self._p.T * self._cost)

        # build objective
        problem = cvx.Problem(objective, constraint)

        return problem

    def step(self, action):

        # TODO(Mathilde): For this first version the environment has no cost (could be in the future to use the grid's constraints
        reward = 0

        if not isinstance(action, np.ndarray):
            action = np.array([action])

        if not self.action_space.contains(action):
            raise ValueError('The action is not in the action space')

        # assign values to the cvxpy parameters
        try:
            self._p_min.value, self._p_max.value, self._cost.value = self.get_bids_actors(action, self._date)
            self._demand.value = self.get_demand(self._date)
        except EmptyDataException:
            done = True
            ob = self._get_obs()
            return ob, reward, done, dict({'date': self._date})

        # solve the problem
        self._opt_problem.solve(verbose=False)
        if self._print_optimality or "optimal" not in self._opt_problem.status:
            raise OptimizationException

        # price_cleared = []
        # for i in range(len(self._p.value)):
        #     if self._p_min.value[i] < self._p.value[i] <= self._p_max.value[i]:
        #         price_cleared.append(self._cost.value[i])

        self._price_cleared = np.max(price_cleared)

        self._date += self._delta_time

        # the state here is the price and capacity cleared
        self._state = np.array([self._p.value[-1]])
        ob = self._get_obs()

        done = False

        return ob, reward, done, dict({'date': self._date})

    def reset(self, start_date=None):
        if start_date is not None:
<<<<<<< HEAD
            self._date = start_date
        else:
            self._date = self._start_date
=======
            self._start_date = start_date
        self._date = self._start_date
>>>>>>> 3144ddbfcb2f8f61a5924ec15ba4d5bad823b4ca
        self._state = np.zeros(self.observation_space.shape[0])
        return self._get_obs()

    def render(self, mode='rgb_array'):
        print('current state:', self._state)

    def _get_obs(self):
        return np.copy(self._state)

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed = seed
        else:
            np.random.seed = seeding.np_random()

    def get_demand(self, date):
        load = self.caiso_get_load(start_at=date, end_at=date + self._delta_time)
        load_list = load['load_MW']
        f = lambda x: x/10
        demand = np.mean(load_list)
        demand = f(demand)
        return demand

    def get_bids_actors(self, action, date):
        gen = self.caiso_get_generation(start_at=date, end_at=date + self._delta_time)
        if gen.empty:

            raise EmptyDataException
        gen_wind_list = gen[gen['fuel_name'] == 'wind']['gen_MW'].values
        gen_solar_list = gen[gen['fuel_name'] == 'solar']['gen_MW'].values
        gen_other_list = gen[gen['fuel_name'] == 'other']['gen_MW'].values
        p_max = np.array([np.mean(gen_wind_list),
                          np.mean(gen_solar_list),
                          np.mean(gen_other_list),
                          100000 + 10000 * (np.mean(gen_wind_list) + np.mean(gen_solar_list) + np.mean(gen_other_list)),
                          max(action[0], 0)])
        p_min = np.zeros(5)
        p_min[-1] = min(action[0], 0)
        cost = np.array([4, 2, 9, 12, action[1]])
        return p_min, p_max, cost

    def caiso_get_generation(self, start_at, end_at):
        start_date_aware = self._timezone.localize(start_at)
        end_date_aware = self._timezone.localize(end_at)
        return self._gen_df[(start_date_aware <= self._gen_df["timestamp"]) &
                           (end_date_aware > self._gen_df["timestamp"])]

    def caiso_get_load(self, start_at, end_at):
        start_date_aware = self._timezone.localize(start_at)
        end_date_aware = self._timezone.localize(end_at)
        return self._dem_df[(start_date_aware <= self._dem_df["timestamp"]) &
                           (end_date_aware > self._dem_df["timestamp"])]
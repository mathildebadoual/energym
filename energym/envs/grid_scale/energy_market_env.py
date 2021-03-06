import gym
from gym import spaces
from gym.utils import seeding
from energym.envs.grid_scale.utils import OptimizationException, EmptyDataException

import pandas as pd
import cvxpy as cvx
import numpy as np
import pytz
import datetime
import logging
import os

logger = logging.getLogger(__name__)


class EnergyMarketEnv(gym.Env):
    def __init__(self, data_dir_name='data'):
        self._num_agents = 5

        self.data_path = os.path.join(os.path.dirname(__file__), data_dir_name)
        self._opt_problem = self.build_opt_problem()
        self._is_test = False
        self._price_benchmark = self.get_price_benchmark(self.data_path)

        self._timezone = pytz.utc
        self._print_optimality = False

        self._delta_time = datetime.timedelta(hours=1)

        # gym variables
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
        # TODO(Mathilde): Add an option for a discrete action space
        self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)

        # the state is the clearance (bool) and the quantity cleared
        self._state = self.reset()

    def set_test(self):
        self._is_test = True

    def get_price_benchmark(self, data_path):
        df = pd.read_csv(os.path.join(data_path, 'prices_benchmark.csv'))
        df = df[df['node'] == 'ALAMT5G_7_N002']
        df1 = df[['dollar_mw', 'hr']]
        df2 = df1.groupby('hr').mean()
        return df2

    def get_start_date(self):
        start_date = pd.to_datetime(self._gen_df['timestamp'].values[0])
        return start_date

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
        # TODO(Mathilde): solve this try/except bad fix
        try:
            self._p_min.value, self._p_max.value, self._cost.value = self.get_bids_actors(action, self._date)
            # if the bid power is negative, it is added to the demand
            self._demand.value = self.get_demand(self._date) + min(action[0], 0)
        except EmptyDataException:
            print('empty data exception')
            done = True
            ob = self._get_obs()
            return ob, reward, done, dict({'date': self._date, 'price_cleared': None, 'ref_price': None})

        ref_price = self._cost.value[0]

        # solve the problem
        try:
            self._opt_problem.solve(verbose=False)
        except cvx.error.SolverError:
            print("ERROR Caught: cvxpy.error.SolverError")
            pass
        if self._print_optimality or "optimal" not in self._opt_problem.status:
            raise OptimizationException

        price_cleared_list = []
        for i in range(len(self._p.value)):
            if self._p_min.value[i] + 1 < self._p.value[i] <= self._p_max.value[i]:
                price_cleared_list.append(self._cost.value[i])

        price_cleared = np.max([0] + price_cleared_list)

        self._date += self._delta_time

        # the state here is the price and capacity cleared
        if action[0] > 0:
            self._state = np.array([self._p.value[-1], self._date.hour])
        else:
            self._state = np.array([action[0], self._date.hour])

        ob = self._get_obs()

        done = False

        return ob, reward, done, dict({'date': self._date, 'price_cleared': price_cleared, 'ref_price': ref_price})

    def reset(self, start_date=None):
        if self._is_test:
            self._gen_df = load_pickles_and_reorder(self.data_path + "/gen_caiso_test.pkl")
            self._dem_df = load_pickles_and_reorder(self.data_path + "/dem_caiso_test.pkl")
        else:
            self._gen_df = load_pickles_and_reorder(self.data_path + "/gen_caiso_train.pkl")
            self._dem_df = load_pickles_and_reorder(self.data_path + "/dem_caiso_train.pkl")

        if start_date is not None:
            self._start_date = start_date
        else:
            self._start_date = pd.to_datetime(self._gen_df['timestamp'].values[0])
        self._date = self._start_date
        self._state = np.array([0, self._date.hour])
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
        if load.empty:
            print('load empty')
            raise EmptyDataException
        load_list = load['load_MW']
        f = lambda x: x / 10
        demand = np.mean(load_list)
        demand = f(demand)
        return demand

    def get_bids_actors(self, action, date):
        """
        This function creates bids for the other actors of the market.
        It uses the data from 3 types of generation from CAISO: solar, wind and other (coal, etc)
        The third agent in the market is an infinite resource of expensive energy. It is used to always match the demand.
        """
        gen = self.caiso_get_generation(start_at=date, end_at=date + self._delta_time)
        if gen.empty:
            print('gen empty')
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
        # p_min[-1] = min(action[0], 0)
        price_benchmark = self._price_benchmark.loc[date.hour].values[0]
        # to avoid using the agent when it is buying energy
        if action[0] <= 0:
            action[1] = 100000
        cost = np.array([price_benchmark + 2, price_benchmark - 1.5,
                         price_benchmark + 7, price_benchmark + 10, action[1]])
        return p_min, p_max, cost

    def caiso_get_generation(self, start_at, end_at):
        """
        Gives the dataframe of the caiso generation data for the period between start_date and end_date
        :param start_at: (datetime) start date
        :param end_at: (datetime) end_date
        :return: (dataframe) gen_df for the required period
        """
        if start_at.tzinfo is None or start_at.tzinfo.utcoffset(start_at) is None:
            start_date_aware = self._timezone.localize(start_at)
        else:
            start_date_aware = start_at
        if end_at.tzinfo is None or end_at.tzinfo.utcoffset(end_at) is None:
            end_date_aware = self._timezone.localize(end_at)
        else:
            end_date_aware = end_at
        return self._gen_df[(start_date_aware <= self._gen_df["timestamp"]) &
                            (end_date_aware > self._gen_df["timestamp"])]

    def caiso_get_load(self, start_at, end_at):
        """
        Gives the dataframe of the caiso load data for the period between start_date and end_date
        :param start_at: (datetime) start date
        :param end_at: (datetime) end_date
        :return: (dataframe) load_df for the required period
        """
        if start_at.tzinfo is None or start_at.tzinfo.utcoffset(start_at) is None:
            start_date_aware = self._timezone.localize(start_at)
        else:
            start_date_aware = start_at
        if end_at.tzinfo is None or end_at.tzinfo.utcoffset(end_at) is None:
            end_date_aware = self._timezone.localize(end_at)
        else:
            end_date_aware = end_at
        return self._dem_df[(start_date_aware <= self._dem_df["timestamp"]) &
                            (end_date_aware > self._dem_df["timestamp"])]


def load_pickles_and_reorder(file_path):
    data = pd.read_pickle(file_path)
    if data.empty:
        raise EmptyDataException
    data = data.sort_values('timestamp')
    return data

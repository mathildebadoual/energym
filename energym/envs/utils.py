import cvxpy as cvx
import gym
import energym
import numpy as np
import pandas as pd
import os
import logging
import datetime


class EmptyDataException(Exception):
    def __init__(self):
        super().__init__()


class OptimizationException(Exception):
    def __init__(self):
        super().__init__()


# The difference between DQN and this expert agent is that it depends on the environment directly

logging.getLogger().setLevel(logging.INFO)

class ExpertAgent(object):
    def __init__(self):
        # this is the environment on which the controller will be applied
        self.env = gym.make('energy_market_battery-v0')

        # we create those environments to get some info (clearing prices + battery dynamic)
        self.market = gym.make('energy_market-v0')
        self.battery = gym.make('battery-v0')

        # to create the prediction prices (perfect forecast)
        self.data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
        self.price_prediction_file_path = os.path.join(self.data_path, "price_prediction.csv")
        self.get_prediction_cleared_prices()
        self.price_prediction_df = self.load_price_predictions()

        # parameters for the online controller
        self.reset_memory_dict()
        self.planning_frequency = 1
        self.time_horizon = 16
        self.max_soe, self.min_soe, self.max_power, self.min_power, self.battery_efficiency = self.battery.get_parameters_battery()

        # create the optimization problem
        self.problem = self.create_optimization_problem()

    def get_prediction_cleared_prices(self):
        # We run the all simulation without the battery (considering we are price take we do not influence the market).
        # This function needs to be called once and then we store the result in a pickle
        if not os.path.exists(self.price_prediction_file_path):
            logging.info('---- Create Prediction Prices ----')
            done = False
            action = np.array([0, 100000])
            i = 0
            price_prediction_dict = {'time_step': [], 'values': []}
            while not done:
                ob, reward, done, info_dict = self.market.step(action)
                price_prediction_dict['values'].append(info_dict['price_cleared'])
                price_prediction_dict['time_step'].append(info_dict['date'])
                if i % 100 == 0 :
                    logging.info('----> Step %s' % (info_dict['date']))
                i += 1
            price_prediction_df = pd.DataFrame.from_dict(price_prediction_dict)
            price_prediction_df.to_csv(self.price_prediction_file_path)

    def load_price_predictions(self):
        logging.info('---- Load Prediction Prices ----')
        price_prediction_df = pd.read_csv(self.price_prediction_file_path)
        return price_prediction_df

    def create_optimization_problem(self):
        # create a generic optimization problem solved for planning
        self.price_predictions_interval = cvx.Parameter(self.time_horizon)
        self.initial_soe = cvx.Parameter()

        self.soe = cvx.Variable(self.time_horizon)
        self.planned_power = cvx.Variable(self.time_horizon)

        opt = cvx.Maximize(self.price_predictions_interval * self.planned_power)

        constraints = [self.soe[0] == self.initial_soe]
        for i in range(self.time_horizon-1):
            constraints += [self.soe[i+1] == self.soe[i] + self.battery_efficiency * self.planned_power[i]]

        constraints += [self.soe <= self.max_soe] + [self.min_soe <= self.soe]
        constraints += [self.planned_power <= self.max_power] + [self.min_power <= self.planned_power]

        return cvx.Problem(opt, constraints)

    def planning(self, step):
        # solve optimization problem from actual time step for a certain horizon
        self.price_prediction_df['time_step'] = pd.to_datetime(self.price_prediction_df['time_step'])

        step = datetime.datetime.strptime(step.strftime("%Y-%m-%d %H:%M:%S.%f"), "%Y-%m-%d %H:%M:%S.%f")
        values_planning_horizon = self.price_prediction_df[self.price_prediction_df['time_step'] >= step]['values']
        
        self.price_predictions_interval.value = np.resize(values_planning_horizon.values[:self.time_horizon], (self.time_horizon,))
        self.initial_soe.value = self.memory_dict['soe'][-1]

        # logging.info('---- Solve Optimization ----')
        self.problem.solve(solver=cvx.CVXOPT, verbose=False)
        # logging.info('---- Status: %s ----' % self.problem.status)
        planned_actions = self.planned_power.value
        return planned_actions

    def running(self, planned_actions):
        # run until time to re-plan, collect same outputs as the RL agent
        done = False
        for i in range(self.time_horizon):
            if i >= self.planning_frequency or done:
                break
            action = [planned_actions[i], self.price_predictions_interval.value[i]]
            ob, reward, done, info_dict = self.env.step(action)
            self.memory_dict['soe'].append(ob[0])
            self.memory_dict['power_cleared'].append(ob[1])
            self.memory_dict['price_bid'].append(self.price_predictions_interval.value[i])
            self.memory_dict['reward'].append(reward)
            self.memory_dict['time_step'].append(info_dict['date'])
            self.memory_dict['done'].append(done)
            self.memory_dict['power_bid'].append(planned_actions[i])
        return done

    def reset_memory_dict(self):
        self.memory_dict = {'soe': [0],
                        'power_cleared': [0],
                        'price_bid': [0],
                        'reward': [0],
                        'done': [0],
                        'time_step': [0],
                        'power_bid': [0],
                        }
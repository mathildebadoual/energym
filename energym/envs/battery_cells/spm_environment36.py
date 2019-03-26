import gym
from gym import spaces
from gym.envs.toy_text import discrete
import numpy as np
from energym.envs.battery_cells.spm_utils36 import *
from control.matlab import *
from energym.envs.battery_cells.paramfile import *

# ==============================================================================
# SPM
# ==============================================================================


DISCRETE = True  # False


class SPM(gym.Env):
    def __init__(self, init_v=3.1):

        # print('Saehong mode ::)')

        # Define input range.
        self.currents = np.linspace(-50, 0, 20)

        # Environment PARAMETERS
        self.discrete = DISCRETE
        self.RSN = p['R_s_n']

        A_n, A_p, B_n, B_p, C_n, C_p, D_n, D_p = spm_plant_obs_mats(p)

        self.sys_n = ss(A_n, B_n, C_n, D_n)  # set up state space
        self.sys_p = ss(A_p, B_p, C_p, D_p)

        csn0, csp0 = init_cs_NMC(p, init_v)  # initial conditions
        # 3V ~= 0.0166 SOCn
        # 4V ~= 0.7333
        self.c_n0 = csn0 * np.ones(p['Nr'] - 1)
        self.c_p0 = csp0 * np.ones(p['Nr'] - 1)
        self.V0 = refPotentialCathode_casadi(csp0 / p['c_s_p_max']) - refPotentialAnode_casadi(csn0 / p['c_s_n_max'])

        # self.currents = np.arange(-50, 50, 0.5)
        # self.action_space = spaces.Discrete(len(self.currents)) #[0, ... , 199]

        self.csn_normal = self.c_n0 / p['c_s_n_max']  # Normalize for env output
        self.csp_normal = self.c_p0 / p['c_s_p_max']  # Normalize for env output
        self.csn = self.c_n0
        self.csp = self.c_p0
        csn_high = np.ones(len(self.csn)) * p['c_s_n_max']
        csp_high = np.ones(len(self.csp)) * p['c_s_p_max']

        cs_high = np.concatenate((csn_high, csp_high))
        # self.observation_space = spaces.Box(np.zeros(len(cs_high)), cs_high, dtype=np.float32)

        self.V = self.V0

        self.css_n = self.c_n0[len(self.c_n0) - 1]
        self.css_p = self.c_p0[len(self.c_p0) - 1]

        delta_r_n = p['delta_r_n']
        c_n = np.concatenate(([self.csn[0]], self.csn, [self.css_n]))
        r = np.arange(0, 1 + delta_r_n / 2, delta_r_n)

        r.reshape(len(r), 1)

        self.SOCn = 3 / (p['c_s_n_max']) * np.trapz(c_n * (r * r), r)
        self.SOC_desired = 0.8

        self.info = dict()
        self.info['SOCn'] = self.SOCn
        self.info['css_n'] = self.css_n
        self.info['css_p'] = self.css_p
        self.info['V'] = self.V

        self.len_csn = len(self.csn)
        self.len_csp = len(self.csp)
        self.M = self.len_csn + self.len_csp

    @property
    def observation_space(self):
        return spaces.Box(low=0, high=49521, shape=(self.M,), dtype=np.float32)

    @property
    def action_space(self):
        if self.discrete:
            return spaces.Discrete(20)
        else:  # continuous case.
            return spaces.Box(dtype=np.float32, low=-50, high=0, shape=(1,))

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):

        """
        action: some form of current input?
        reward: based on the increase in SoC?
        next state: output the new state based on the action


        """

        is_done = False
        # if not self.action_space.contains(action):
        # 	print('invalid action!')
        # 	return None

        if self.discrete:
            action = self.currents[action]
        else:
            action = action
            action = np.clip(action, a_min=self.action_space.low, a_max=self.action_space.high)[0]

        I = np.array([action, action])
        t = np.array([0, 1])
        # lsim takes only arrays, so even though we only want one time step,
        # we need to run the simulation for 2 time steps
        test_cssn, test_t, test_cnx = lsim(self.sys_n, I, t, self.csn)

        self.csn = test_cnx[1, :]  # get second row

        self.css_n = test_cssn[1]  # get second item

        test_cssp, test_t, test_cpx = lsim(self.sys_p, I, t, self.csp)

        self.csp = test_cpx[1, :]  # get second row

        self.css_p = test_cssp[1]  # get second item

        # calculate the current SOCn
        delta_r_n = p['delta_r_n']
        c_n = np.concatenate(([self.csn[0]], self.csn, [self.css_n]))
        r = np.arange(0, 1 + delta_r_n / 2, delta_r_n)

        r.reshape(len(r), 1)

        self.SOCn = 3 / (p['c_s_n_max']) * np.trapz(c_n * (r * r), r)

        if self.SOCn >= 0.8:
            is_done = True

        self.V = nonlinear_SPM_Voltage(p, self.css_n, self.css_p, 1000, 1000, 1000, action)

        reward = -((self.SOCn - self.SOC_desired) / self.SOC_desired) ** 2  # reward = - || 0.8 - curr_SoC||

        # reward = - (np.linalg.norm(self.SOCn - self.SOC_desired, ord=2)/self.SOC_desired)

        # reward = - (np.linalg.norm(self.SOCn-self.SOC_desired, ord=2)/self.SOC_desired
        # update the info dictionary

        self.info['SOCn'] = self.SOCn
        self.info['css_n'] = self.css_n
        self.info['css_p'] = self.css_p
        self.info['V'] = self.V

        return np.concatenate((self.csn / p['c_s_n_max'], self.csp / p['c_s_p_max'])), reward, is_done, self.info

    def render(self):
        pass

    def reset(self):

        self.csn = self.c_n0
        self.csp = self.c_p0
        csn_high = np.ones(len(self.csn)) * p['c_s_n_max']
        csp_high = np.ones(len(self.csp)) * p['c_s_p_max']

        cs_high = np.concatenate((csn_high, csp_high))
        # self.observation_space = spaces.Box(np.zeros(len(cs_high)), cs_high, dtype=np.float32)

        self.V = self.V0

        self.css_n = self.c_n0[len(self.c_n0) - 1]
        self.css_p = self.c_p0[len(self.c_p0) - 1]

        delta_r_n = p['delta_r_n']
        c_n = np.concatenate(([self.csn[0]], self.csn, [self.css_n]))
        r = np.arange(0, 1 + delta_r_n / 2, delta_r_n)

        r.reshape(len(r), 1)

        self.SOCn = 3 / (p['c_s_n_max']) * np.trapz(c_n * (r * r), r)

        self.info = dict()
        self.info['SOCn'] = self.SOCn
        self.info['css_n'] = self.css_n
        self.info['css_p'] = self.css_p
        self.info['V'] = self.V

        return np.concatenate((self.csn / p['c_s_n_max'], self.csp / p['c_s_p_max']))

# #
# # # SIMULATIONS
# new_env = SPM()
#
# SOCn_list = [new_env.SOCn]
# V_list = [new_env.V]
# css_n_list = [new_env.css_n]
#
# done = False
# while not done:
#     # in this example we give a constant current of -5 Amps
#     csn, reward, done, info = new_env.step(-50)
#     css_n_list.append(info['css_n'])
#     SOCn_list.append(info['SOCn'])
#     V_list.append(info['V'])
#
# # print(V_list)
#
# plt.figure()
# plt.plot(V_list)
# plt.title("Voltage over time")
# plt.ylabel("Voltage")
# plt.xlabel("time")
# plt.show()
#
# plt.figure()
# plt.plot(SOCn_list)
# plt.title("State of Charge over time")
# plt.ylabel("SOC")
# plt.xlabel("time")
# plt.show()
#
# plt.figure()
# plt.plot(css_n_list)
# plt.title("Anode Surface Concentration over time")
# plt.ylabel("css_n")
# plt.xlabel("time")
# plt.show()

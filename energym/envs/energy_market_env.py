import gym
from gym import Env, error, spaces, utils
from gym.utils import seeding

import logging
logger = logging.getLogger(__name__)


class EnergyMarketEnv(Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.reset()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,))
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(2,))

    def step(self, action):
        return ob, reward, done, dict()

    def reset(self):
        self._state = np.array([0, 0], dtype=np.float32)
        return self._get_obs()

    def render(self):
        print('current state:', self._state)

    def _get_obs(self):
        return np.copy(self._state)

    def seed(self, seed):
        np.random.seed = seed

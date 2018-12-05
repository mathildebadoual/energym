import unittest
import gym
import energym

from energym.envs.utils import OptimizationException, EmptyDataException, ExpertAgent


class TestExceptions(unittest.TestCase):
	def setUp(self):
		pass

	def test_optimization_exception(self):
		pass

	def test_empty_data_exception(self):
		pass


class TestExpertAgent(unittest.TestCase):
	def setUp(self):
		self.expert_agent = ExpertAgent()

	def test_expert_agent(self):
		pass
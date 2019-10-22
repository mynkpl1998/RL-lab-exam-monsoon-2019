import numpy as np

class epsilonGreedy:

	def __init__(self, epsilon=0.1):
		self.epsilon = epsilon

	def getAction(self, Qvector, env):
		
		if np.random.rand() <= self.epsilon:
			return env.action_space.sample()
		else:
			return np.argmax(Qvector)

	def getGreedyAction(self, Qvector):
		return np.argmax(Qvector)
	
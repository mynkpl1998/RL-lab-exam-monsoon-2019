import numpy as np

class qLearner:

	def __init__(self, numActs, envObj, explorationPolicy, discountFactor=0.9, learningRate=0.01, initValue=0.0, numIters=10000):
		self.initValue = initValue
		self.discountFactor = discountFactor
		self.learningRate = learningRate
		self.numActs = numActs
		self.numIters = numIters
		self.envObj = envObj
		self.explorationPolicy = explorationPolicy

	def resetDict(self):
		self.qDict = {}

	def buildDummyActionSpace(self):
		return np.ones(self.numActs).copy() * self.initValue

	def addToDict(self, state):
		if state not in self.qDict.keys():
			self.qDict[state] = self.buildDummyActionSpace()

	def execute(self, renderPolicy=False):
		prevState = self.envObj.reset()
		done = False
		count = 0
		if renderPolicy:
			print("Start State : ")
			self.envObj.render()
		
		cumReward = 0.0
		while not done:
			action = self.explorationPolicy.getGreedyAction(self.qDict[prevState])
			if renderPolicy:
				print("Action : ", self.envObj.inv_act_map[action])
			nextState, reward, done, _ = self.envObj.step(action)
			cumReward += reward
			if renderPolicy:
				self.envObj.render()
			prevState = nextState
			count += 1

			if count > 100:
				break

		return cumReward

	def learn(self):
		totalSteps = 0
		self.resetDict()

		self.cumRewards = []
		self.episodes = []
		episode = 0
		self.tdErrorList = []

		while totalSteps < self.numIters:
			prevState = self.envObj.reset()
			self.addToDict(prevState)
			cumReward = 0.0
			done = False
			while not done:
				action = self.explorationPolicy.getAction(self.qDict[prevState], self.envObj)
				nextState, reward, done, _ = self.envObj.step(action)
				self.addToDict(nextState)
				
				# Q-value update step
				bootstrappedTarget = reward + self.discountFactor * np.max(self.qDict[nextState])
				currentEstimate = self.qDict[prevState][action]
				tdError = bootstrappedTarget - currentEstimate
				self.tdErrorList.append(np.absolute(tdError))
				cumReward += reward
				self.qDict[prevState][action] += self.learningRate * tdError
				prevState = nextState

				totalSteps += 1

				if done:
					episode += 1
					self.episodes.append(episode)
					self.cumRewards.append(cumReward)

		print("Total States in Q-dict : ", len(self.qDict))
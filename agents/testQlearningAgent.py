from qLearningAgent import qLearner
import sys, os
sys.path.append(os.getcwd()[:-7] + "/envs")
sys.path.append(os.getcwd()[:-7] + "/explorer")
from clif import Clif
from epsilonGreedy import epsilonGreedy


env = Clif(10, 5)
explorer = epsilonGreedy(0.1)
agent = qLearner(env.action_space.n, env, explorer, numIters=100000)

agent.learn()
agent.execute()

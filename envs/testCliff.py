from clif import Clif

env = Clif(10, 5)

for i in range(0, 10):
	print(env.reset())
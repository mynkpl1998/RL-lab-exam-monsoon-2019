import numpy as np
import gym
from gym.spaces import Box, Discrete

class Clif():
  
  def __init__(self,corr_len=10, corr_height=3, useObstacles=False):
    
    if corr_len < 3:
      print("Length should be greater than 3")
      sys.exit(-1)
     
    if corr_height < 2:
      print("Height should be greater than 1 ")
      sys.exit(-2)
    
    self.corridor_length = corr_len
    self.corr_height = corr_height
    
    self.observation_space = Box(-float("inf"), float("inf"), shape=(self.corridor_length * corr_height,), dtype=np.float32)
    
    self.init_actions_map()

    self.obstacles = self.genObstacles(self.corr_height-2, self.corr_height-2, self.corridor_length-2)
    self.useObstacles = useObstacles
   
  
  def init_actions_map(self):
    
    self.act_map = {}
    self.act_map["UP"] = 0
    self.act_map["DOWN"] = 1
    self.act_map["LEFT"] = 2
    self.act_map["RIGHT"] = 3
    
    self.inv_act_map = {v: k for k, v in self.act_map.items()}
    
    self.action_space = Discrete(len(self.act_map))
   
  def getObs(self):  
    tmp = self.grid.copy()
    tmp[self.agent_loc[0], self.agent_loc[1]] = 2
    
    return tmp.copy()

  def obs2str(self, obs):    
    return np.array2string(obs)

  def render(self):
    print(self.getObs())

  def genObstacles(self, numObstacles, xHigh, yHigh):
    obstacles = []
    for i in range(0, numObstacles):
      tup = (i+1, np.random.randint(1, yHigh))
      obstacles.append(tup)
    return obstacles
  
  def reset(self):
    self.grid = np.zeros((self.corr_height, self.corridor_length), dtype=np.int32)
    self.grid[self.corr_height - 1, 1: self.corridor_length-1] = 1
    self.agent_loc = [self.corr_height - 1, 0]
    self.goal_loc = [self.corr_height - 1, self.corridor_length-1]    
    # Populate grid with obstacles
    

    if self.useObstacles:
      for obstacle in self.obstacles:
        #print(obstacle)
        self.grid[obstacle[0], obstacle[1]] = 1
    
    return self.obs2str(self.getObs())
  
  
  def step(self, action):
    
    if action == 0:
     
      self.agent_loc[0] -=1
      
      if self.agent_loc[0] < 0:
        self.agent_loc[0] = 0
        #done = True
        #reward = -euclidean(self.agent_loc, self.goal_loc) / euclidean([0,0], self.goal_loc)
      
    elif action == 1:
      
      self.agent_loc[0] +=1
      
      if self.agent_loc[0] > self.corr_height-1:
        self.agent_loc[0] = self.corr_height-1
        #done = True
        #reward = -euclidean(self.agent_loc, self.goal_loc) / euclidean([0,0], self.goal_loc)
      
    elif action == 2:
      
      self.agent_loc[1] -= 1
      
      if self.agent_loc[1] < 0:
        self.agent_loc[1] = 0
        #done = True
        #reward = -euclidean(self.agent_loc, self.goal_loc) / euclidean([0,0], self.goal_loc)
      
    else:
      
      self.agent_loc[1] +=1
      
      if self.agent_loc[1] > self.corridor_length - 1:
        self.agent_loc[1] = self.corridor_length - 1
    
    reward = -1
    
    if self.grid[self.agent_loc[0], self.agent_loc[1]] == 1:
      reward = -100
      self.agent_loc = [self.corr_height - 1, 0]
    
    if self.agent_loc == self.goal_loc:
      done = True
    else:
      done = False
    
    return (self.obs2str(self.getObs()), reward, done, {})
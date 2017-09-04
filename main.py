import gym
import numpy as np

import train

env = gym.make('BipedalWalker-v2')

MAX_EPISODES = 100
MAX_STEPS = 50
S_DIM = env.observation_space.shape
A_DIM = env.action_space.shape
A_MAX = env.action_space.high

print ' State Dimensions :- ', S_DIM
print ' Action Dimensions :- ', A_DIM
print ' Action Max :- ', A_MAX

trainer = train.Trainer(S_DIM, A_DIM, A_MAX)

for _ep in range(MAX_EPISODES):
	observation = env.reset()
	for r in range(MAX_STEPS):
		env.render()
		
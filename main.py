import gym
import numpy as np
import torch
from torch.autograd import Variable

import train
import buffer

env = gym.make('BipedalWalker-v2')

MAX_EPISODES = 1
MAX_STEPS = 50
MAX_BUFFER = 256
S_DIM = env.observation_space.shape[0]
A_DIM = env.action_space.shape[0]
A_MAX = env.action_space.high[0]

print ' State Dimensions :- ', S_DIM
print ' Action Dimensions :- ', A_DIM
print ' Action Max :- ', A_MAX

ram = buffer.MemoryBuffer(MAX_BUFFER)
trainer = train.Trainer(S_DIM, A_DIM, A_MAX, ram)

for _ep in range(MAX_EPISODES):
	observation = env.reset()
	for r in range(MAX_STEPS):
		env.render()
		state = np.float32(observation)

		# get action based on observation
		# convert numpy array to torch variable and convert the obtained value back
		action = trainer.actor.forward(Variable(torch.from_numpy(state))).data.numpy()
		new_observation , reward, done, info = env.step(action)

		observation = new_observation
		if done:
			new_state = None
		else:
			new_state = np.float32(observation)

		# push this exp in ram
		ram.add(state, action, reward, new_state)

		# perform optimization
		trainer.optimize()
		if done:
			break

print 'Completed episodes'

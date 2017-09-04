import gym
import numpy as np

import train
import buffer

env = gym.make('BipedalWalker-v2')

MAX_EPISODES = 100
MAX_STEPS = 50
MAX_BUFFER = 256
S_DIM = env.observation_space.shape
A_DIM = env.action_space.shape
A_MAX = env.action_space.high

print ' State Dimensions :- ', S_DIM
print ' Action Dimensions :- ', A_DIM
print ' Action Max :- ', A_MAX

ram = buffer.MemoryBuffer(MAX_BUFFER)
trainer = train.Trainer(S_DIM, A_DIM, A_MAX, ram)

for _ep in range(MAX_EPISODES):
	prev_observation = env.reset()
	new_observation = prev_observation
	for r in range(MAX_STEPS):
		env.render()
		state = new_observation - prev_observation

		# get action based on observation
		action = trainer.actor.forward(state)
		observation_ , reward, done, info = env.step(action)

		prev_observation = new_observation
		new_observation = observation_
		if done:
			new_state = None
		else:
			new_state = new_observation - prev_observation

		# push this exp in ram
		ram.add(state, action, reward, new_state)

		# perform optimization
		trainer.optimize()
		if done:
			break

print 'Completed episodes'

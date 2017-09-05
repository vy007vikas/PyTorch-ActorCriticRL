import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

import model

BATCH_SIZE = 64
LEARNING_RATE = 0.001
SIGMA = 0.05
GAMMA = 0.1


def gaussian_policy(mean, dev=0.1, lim=1.0):
	sampled_action = np.random.normal(mean, dev)
	sampled_action = np.maximum(sampled_action, -lim * np.ones_like(sampled_action))
	sampled_action = np.minimum(sampled_action, lim * np.ones_like(sampled_action))
	return sampled_action.astype(np.float32)


class Trainer:

	def __init__(self, state_dim, action_dim, action_max, ram):
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.action_max = action_max
		self.ram = ram
		self.iter = 0

		self.actor = model.Actor(self.state_dim, self.action_dim, self.action_max)
		self.critic = model.Critic(self.state_dim, self.action_dim)

		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),LEARNING_RATE)

	def optimize(self):
		s,a,r,s1 = self.ram.sample(BATCH_SIZE)

		s = Variable(torch.from_numpy(s))
		a = Variable(torch.from_numpy(a))
		r = Variable(torch.from_numpy(r))
		s1 = Variable(torch.from_numpy(s1))

		# print 'S :- ', s.size()
		# print 'A :- ', a.size()
		# print 'R :- ', r.size()
		# print 'S1 :- ', s1.size()

		# Sample action using eps-greedy critic policy
		a1 = self.actor.forward(s1).data.numpy()
		sampled_action = Variable(torch.from_numpy(gaussian_policy(a1, SIGMA, self.action_max)))

		# obtain next state value function in shape of [n,1] and convert to [n,]
		next_val = torch.squeeze(self.critic.forward(s1, sampled_action))

		y_actual = r + GAMMA*next_val
		y_pred = self.critic.forward(s,a)

		# print y_actual , y_pred

		# compute huber loss
		loss = F.smooth_l1_loss(y_pred, y_actual)
		self.critic_optimizer.zero_grad()
		loss.backward()
		self.critic_optimizer.step()

		print 'Iteration :- ', self.iter, ' Loss :- ', loss.data.numpy()[0]

		self.iter += 1

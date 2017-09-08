import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import math

import model

BATCH_SIZE = 64
LEARNING_RATE = 0.001
SIGMA = 0.05
GAMMA = 0.999


def gaussian_policy(mean, dev=0.1, lim=1.0):
	"""
	samples a random variable from the input gaussian distribution
	:param mean: mean (numpy array)
	:param dev: standard deviation (sigma) (numpy array)
	:param lim: used to limit the action in [-limit,limit] region (float)
	:return: sampled action (numpy array)
	"""
	sampled_action = np.random.normal(mean, dev)
	sampled_action = np.maximum(sampled_action, -lim * np.ones_like(sampled_action))
	sampled_action = np.minimum(sampled_action, lim * np.ones_like(sampled_action))
	return sampled_action.astype(np.float32)


def gaussian_distribution(x, mean, dev=0.1):
	"""
	provides value of gaussian probability at the input place
	:param x: input place (Torch Variable: [n,m] )
	:param mean: mean (Torch Variable: [n,m] )
	:param dev: standard deviation (sigma) (Torch Variable: [n,m] )
	:return: probability (Torch Variable: [n,m] )
	"""
	num = (-1*(x-mean).pow(2))/(2*dev.pow(2))
	num = num.exp()
	den = (math.sqrt(2*math.pi))*dev
	return num/den


class Trainer:

	def __init__(self, state_dim, action_dim, action_max, ram):
		"""
		:param state_dim: Dimensions of state (int)
		:param action_dim: Dimension of action (int)
		:param action_max: used to limit the action in [-action_max, action_max] region
		:param ram: replay memory buffer object
		:return:
		"""
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.action_max = action_max
		self.ram = ram
		self.iter = 0

		self.actor = model.Actor(self.state_dim, self.action_dim, self.action_max)
		self.critic = model.Critic(self.state_dim, self.action_dim)

		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),LEARNING_RATE)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),LEARNING_RATE)

	def get_exploitation_action(self, state):
		"""
		gets the action sampled from distribution obtained from actor
		:param state: state (numpy array)
		:return: sampled action (numpy array)
		"""
		state = Variable(torch.from_numpy(state))

		mean, _ = self.actor.forward(state)
		return mean.data.numpy()

	def get_exploration_action(self, state):
		"""
		gets the action sampled from distribution obtained from actor added with exploration noise
		:param state: state (numpy array)
		:return: sampled action (numpy array)
		"""
		state = Variable(torch.from_numpy(state))

		mean, sigma = self.actor.forward(state)
		return gaussian_policy(mean.data.numpy(), sigma.data.numpy(), self.action_max)

	def optimize(self):
		"""
		Samples a random batch from replay memory and performs optimization
		:return:
		"""
		s1,a1,r1,s2 = self.ram.sample(BATCH_SIZE)

		s1 = Variable(torch.from_numpy(s1))
		a1 = Variable(torch.from_numpy(a1))
		r1 = Variable(torch.from_numpy(r1))
		s2 = Variable(torch.from_numpy(s2))

		# Use exploitation policy here for loss evaluation
		a2 = self.get_exploitation_action(s2.data.numpy())
		a2 = Variable(torch.from_numpy(a2))
		# obtain next state value function in shape of [n,1] and convert to [n,]
		next_val = torch.squeeze(self.critic.forward(s2, a2))

		y_expected = r1 + GAMMA*next_val
		y_predicted = torch.squeeze(self.critic.forward(s1, a1))

		# print a1
		# print y_predicted

		# compute critic loss, and update the critic
		loss_critic = F.smooth_l1_loss(y_predicted, y_expected)
		self.critic_optimizer.zero_grad()
		loss_critic.backward()
		self.critic_optimizer.step()

		# compute actor loss and update it
		# compute action pdf for s1, and find probability at a1
		pred_mean_s1, pred_sigma_s1 = self.actor.forward(s1)
		probs = gaussian_distribution(a1, pred_mean_s1, pred_sigma_s1)
		# sum the log of probs along one row to get the net log probability of action_dim actions
		score_function = torch.squeeze(torch.sum(torch.log(probs), dim=1))
		# calling again to create a new graph, because the previous one had gradients already applied
		y_predicted = torch.squeeze(self.critic.forward(s1, a1))
		# loss = sum [ -log( pi(a1|s1) ) * Q(s1,a1) ]
		loss_actor = -torch.sum(score_function*y_predicted)
		self.actor_optimizer.zero_grad()
		loss_actor.backward()
		self.actor_optimizer.step()

		if self.iter % 50 == 0:
			print 'Iteration :- ', self.iter, ' Loss_actor :- ', loss_actor.data.numpy(),\
				' Loss_critic :- ', loss_critic.data.numpy()
		self.iter += 1

from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import math

import model

BATCH_SIZE = 64
LEARNING_RATE = 0.00001
SIGMA = 0.05
GAMMA = 0.999




class Trainer:

	def __init__(self, state_dim, action_dim, ram):
		"""
		:param state_dim: Dimensions of state (int)
		:param action_dim: Dimension of action (int)
		:param ram: replay memory buffer object
		:return:
		"""
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.ram = ram
		self.iter = 0

		self.actor = model.Actor(self.state_dim, self.action_dim)
		self.critic = model.Critic(self.state_dim, self.action_dim)

		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),LEARNING_RATE)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),LEARNING_RATE)

	def get_exploitation_action(self, state):
		"""
		gets the action obtained from actor
		:param state: state (numpy array)
		:return: sampled action (numpy array)
		"""
		state = Variable(torch.from_numpy(state))

		action = self.actor.forward(state)
		return action.data.numpy()

	def get_exploration_action(self, state):
		"""
		gets the action from actor added with exploration noise
		:param state: state (numpy array)
		:return: sampled action (numpy array)
		"""
		state = Variable(torch.from_numpy(state))

		action = self.actor.forward(state)
		# new_action = action.data.numpy() + ......

		return Variable(torch.from_numpy(new_action))

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

		# ---------------------- optimize critic
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

		# ----------------------- optimize actor
		# compute action for s1, and find dQ(s,a)/da
		pred_a1 = self.actor.forward(s1)
		value = torch.squeeze(self.critic.forward(s2, a2))
		value.backward()



		# sum the log of probs along one row to get the net log probability of action_dim actions
		score_function = torch.squeeze(torch.sum(torch.log(probs), dim=1))
		# calling again to create a new graph, because the previous one had gradients already applied
		y_predicted = torch.squeeze(self.critic.forward(s1, a1))
		# loss = sum [ -log( pi(a1|s1) ) * Q(s1,a1) ]
		loss_actor = -torch.sum(score_function*y_predicted)
		self.actor_optimizer.zero_grad()
		loss_actor.backward()
		self.actor_optimizer.step()

		if self.iter % 100 == 0:
			print 'Iteration :- ', self.iter, ' Loss_actor :- ', loss_actor.data.numpy(),\
				' Loss_critic :- ', loss_critic.data.numpy()
		self.iter += 1

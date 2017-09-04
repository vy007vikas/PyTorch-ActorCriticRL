import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import model

BATCH_SIZE = 64
LEARNING_RATE = 0.001
SIGMA = 0.05


def eps_gaussian_policy(mean, dev = 0.1, lim = 1):
	sampled_action = np.random.normal(mean,dev)
	sampled_action = np.maximum(sampled_action, -lim * np.ones_like(sampled_action))
	sampled_action = np.minimum(sampled_action, lim * np.ones_like(sampled_action))
	return sampled_action


class Trainer:

	def __init__(self, state_dim, action_dim, action_max, ram):
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.action_max = action_max
		self.ram = ram

		self.actor = model.Actor(state_dim, action_max)
		self.critic = model.Critic(state_dim, action_dim)

	def optimize(self):
		s,a,r,s1 = self.ram.sample(BATCH_SIZE)

		# Sample action using eps-greedy critic policy
		a1 = self.actor.forward(s1)
		sampled_action = eps_gaussian_policy(a1, SIGMA, self.action_max)
		y_pred = r + self.critic.forward(s1, sampled_action)


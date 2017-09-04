import torch
import torch.nn as nn
import torch.nn.functional as F

import model


class Trainer:

	def __init__(self, state_dim, action_dim, action_max):

		self.state_dim = state_dim
		self.action_dim = action_dim
		self.action_max = action_max

		self.actor = model.Actor(state_dim, action_max)
		self.critic = model.Critic(state_dim, action_dim)


	def update(self, input):



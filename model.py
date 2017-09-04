import torch
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.module):

	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()

		self.state_dim = state_dim
		self.action_dim = action_dim

		self.fcs1 = nn.Linear(state_dim,128)
		self.fca1 = nn.Linear(action_dim,128)

		self.fc2 = nn.Linear(256,128)
		self.fc3 = nn.Linear(128,1)

	def forward(self, state, action):
		s1 = F.relu(self.fcs1(state))
		a1 = F.relu(self.fca1(action))
		x = torch.cat((s1,a1),dim=1)

		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))

		return x


class Actor(nn.module):

	def __init__(self, state_dim, action_max):
		super(Actor, self).__init__()

		self.state_dim = state_dim
		self.action_max = action_max

		self.fc1 = nn.Linear(state_dim,128)
		self.fc2 = nn.Linear(128,64)
		self.fc3 = nn.Linear(64,1)

	def forward(self, state):
		x = F.relu(self.fc1(state))
		x = F.relu(self.fc2(x))
		x = F.tanh(self.fc3(x))

		x = torch.mul(x,self.action_max)

		return x




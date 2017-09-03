import numpy as np
import random
from collections import deque


class MemoryBuffer:

	def __init__(self, size):
		self.buffer = deque()
		self.maxSize = size
		self.len = 0

	def sample(self, count):
		batch = []
		count = max(count, self.len)
		batch = random.sample(self.buffer, count)

		s_arr = np.array([arr[0] for arr in batch])
		a_arr = np.array([arr[1] for arr in batch])
		r_arr = np.array([arr[2] for arr in batch])
		s1_arr = np.array([arr[3] for arr in batch])
		t_arr = np.array([arr[4] for arr in batch])

		return s_arr, a_arr, r_arr, s1_arr, t_arr

	def len(self):
		return self.len

	def add(self, s, a, r, s1, t):
		value = (s,a,r,s1,t)
		self.buffer.append(value)
		if self.len < self.maxSize:
			self.len += 1
		else :
			self.buffer.popleft()

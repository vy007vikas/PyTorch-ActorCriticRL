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
		

	def len(self):
		return self.len

	def add(self, s, a, r, s1, t):
		value = (s,a,r,s1,t)
		self.buffer.append(value)
		if self.len < self.maxSize:
			self.len += 1
		else :
			self.buffer.popleft()

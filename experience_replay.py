# experience_replay.py
import random
from collections import deque

class ExperienceReplay:
    def __init__(self, capacity=10000):
        self.memory = deque(maxlen=capacity)

    def push(self, experience):
        self.memory.append(experience)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

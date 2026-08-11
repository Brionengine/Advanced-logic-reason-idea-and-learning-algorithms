# environment.py
from __future__ import annotations

try:
    import numpy as np
except ImportError:  # optional dependency: pip install numpy
    np = None

class Environment:
    def __init__(self):
        self.state_size = 10  # Example state size
        self.action_size = 4  # Example action size
        self.max_steps = 100

    def reset(self):
        self.current_step = 0
        self.state = np.random.rand(self.state_size)
        return self.state

    def step(self, action):
        self.current_step += 1
        # Placeholder logic for state transition
        next_state = np.random.rand(self.state_size)
        # Placeholder logic for reward
        reward = np.random.randn()
        done = self.current_step >= self.max_steps
        return next_state, reward, done

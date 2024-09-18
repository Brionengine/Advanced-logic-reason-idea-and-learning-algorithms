# agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

class Agent(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64, learning_rate=1e-3, batch_size=64):
        super(Agent, self).__init__()
        self.batch_size = batch_size
        self.state_size = state_size
        self.action_size = action_size

        # Define neural network architecture
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

    def forward(self, state):
        return self.network(state)

    def select_action(self, state, epsilon=0.1):
        if random.random() < epsilon:
            # Exploration: choose a random action
            return random.randrange(self.action_size)
        else:
            # Exploitation: choose the best action based on current policy
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action_values = self.forward(state_tensor)
            return torch.argmax(action_values).item()

    def learn(self, experiences, gamma=0.99):
        states, actions, rewards, next_states, dones = zip(*experiences)
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones).unsqueeze(1)

        # Compute Q targets
        with torch.no_grad():
            Q_targets_next = self.forward(next_states).max(1)[0].unsqueeze(1)
            Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Compute Q expected
        Q_expected = self.forward(states).gather(1, actions)

        # Compute loss
        loss = self.loss_fn(Q_expected, Q_targets)

        # Minimize loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

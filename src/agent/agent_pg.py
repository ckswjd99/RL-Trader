import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque

class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.softmax(self.fc2(x), dim=-1)

class AgentPG:
    def __init__(self, state_size, lr=0.001):
        self.state_size = state_size
        self.action_size = 3
        self.model = PolicyNetwork(state_size, self.action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.memory = []

    def act(self, state):
        state = torch.FloatTensor(state)
        probs = self.model(state)
        action = np.random.choice(self.action_size, p=probs.detach().numpy()[0])
        return action

    def remember(self, state, action, reward):
        self.memory.append((state, action, reward))

    def train(self, gamma=0.99):
        R = 0
        rewards = []
        for _, _, r in reversed(self.memory):
            R = r + gamma * R
            rewards.insert(0, R)
        rewards = torch.FloatTensor(rewards)
        loss = 0
        for (state, action, _), R in zip(self.memory, rewards):
            state = torch.FloatTensor(state)
            probs = self.model(state)
            log_prob = torch.log(probs[0][action])
            loss += -log_prob * R
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.memory = []
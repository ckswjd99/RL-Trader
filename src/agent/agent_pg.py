import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from agent.agent import Agent

class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Linear(8, action_size)
        )

    def forward(self, x):
        return torch.softmax(self.fc(x), dim=-1)

class AgentPG(Agent):
    def __init__(self, state_size, lr=0.001):
        super().__init__(state_size, action_size=3)
        self.model = PolicyNetwork(state_size, self.action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.memory = []

    def act(self, state):
        state = torch.FloatTensor(state)
        probs = self.model(state)

        probs_np = probs.detach().numpy()[0]
        if np.isnan(probs_np).any():
            print("[ERROR] NaN detected in action probabilities!")
            print("probs =", probs_np)
            print("raw logits =", self.model(state).detach().numpy()[0])  # 또는 policy_output
            exit(1)

        action = np.random.choice(self.action_size, p=probs_np)
        return action

    def remember(self, state, action, reward, next_state=None, done=None):
        self.memory.append((state, action, reward))

    def train_step(self, batch_size):
        # PG는 step마다 별도 학습 없음
        pass

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
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class QNet(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 8)
        self.fc4 = nn.Linear(8, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)

class AgentDeepSARSA:
    def __init__(self, state_size, lr=0.001):
        self.state_size = state_size
        self.action_size = 3
        self.memory = deque(maxlen=1000)
        self.inventory = []
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = QNet(state_size, self.action_size).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()

    def expReplay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        mini_batch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in mini_batch:
            state_t = torch.FloatTensor(state).to(self.device)
            next_state_t = torch.FloatTensor(next_state).to(self.device)
            target = reward
            if not done:
                with torch.no_grad():
                    next_action = torch.argmax(self.model(next_state_t)).item()
                    target = reward + self.gamma * self.model(next_state_t)[0][next_action].item()
            target_f = self.model(state_t)
            target_val = target_f.clone().detach()
            target_val[0][action] = target

            self.optimizer.zero_grad()
            loss = self.criterion(target_f[0][action], torch.tensor(target, dtype=torch.float32).to(self.device))
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train(self):
        pass
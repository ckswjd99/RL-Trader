import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from agent.agent import Agent

class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU()
        )
        self.actor = nn.Linear(64, action_size)
        self.critic = nn.Linear(64, 1)

    def forward(self, x):
        x = self.fc(x)
        return torch.softmax(self.actor(x), dim=-1), self.critic(x)

class AgentA3C(Agent):
    def __init__(self, state_size, lr=0.001):
        super().__init__(state_size, action_size=3)
        self.model = ActorCritic(state_size, self.action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.memory = []
        self.inventory = []

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_step(self, batch_size):
        # A3C는 step마다 별도 학습 없음
        pass

    def train(self, gamma=0.99):
        for state, action, reward, next_state, done in self.memory:
            state_t = torch.FloatTensor(state)
            next_state_t = torch.FloatTensor(next_state)
            probs, value = self.model(state_t)
            _, next_value = self.model(next_state_t)
            target = reward + (0 if done else gamma * next_value.item())
            advantage = target - value.item()

            log_prob = torch.log(probs[0][action])
            actor_loss = -log_prob * advantage
            critic_loss = (target - value) ** 2

            loss = actor_loss + critic_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self.memory = []

    def act(self, state):
        state = torch.FloatTensor(state)
        probs, _ = self.model(state)
        action = np.random.choice(self.action_size, p=probs.detach().numpy()[0])
        return action
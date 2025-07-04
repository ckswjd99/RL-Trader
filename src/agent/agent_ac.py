import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from agent.agent import Agent

class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Linear(8, action_size),
            nn.Softmax(dim=-1)
        )
    def forward(self, x):
        return self.fc(x)

class Critic(nn.Module):
    def __init__(self, state_size):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Linear(8, 1)  # Critic outputs a single value
        )
    def forward(self, x):
        return self.fc(x)

class AgentAC(Agent):
    def __init__(self, state_size, lr=0.001):
        super().__init__(state_size, action_size=3)
        self.actor = Actor(state_size, self.action_size)
        self.critic = Critic(state_size)
        self.optimizerA = optim.Adam(self.actor.parameters(), lr=lr)
        self.optimizerC = optim.Adam(self.critic.parameters(), lr=lr)
        self.memory = []
        self.inventory = []

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        state = torch.FloatTensor(state)
        probs = self.actor(state)

        # detach, numpy 변환, 차원 확장
        prob_np = probs.detach().numpy().flatten()

        # 안전장치 1: NaN 또는 inf 확인 → fallback
        if not np.all(np.isfinite(prob_np)):
            return np.random.choice(self.action_size)

        # 안전장치 2: 음수값 제거 → clip 후 정규화
        prob_np = np.clip(prob_np, 0, 1)
        total = prob_np.sum()
        if total == 0 or not np.isfinite(total):
            prob_np = np.ones(self.action_size) / self.action_size
        else:
            prob_np = prob_np / total

        # 안전장치 3: 최종 확률 합산 확인
        if not np.isclose(prob_np.sum(), 1.0, atol=1e-3):
            prob_np = np.ones(self.action_size) / self.action_size

        return np.random.choice(self.action_size, p=prob_np)


    def train_step(self, batch_size):
        # AC는 step마다 별도 학습 없음
        pass

    def train(self, gamma=0.99):
        for state, action, reward, next_state, done in self.memory:
            state_t = torch.FloatTensor(state)
            next_state_t = torch.FloatTensor(next_state)
            value = self.critic(state_t)
            next_value = self.critic(next_state_t)
            target = reward + (0 if done else gamma * next_value.item())
            advantage = target - value.item()

            # Actor loss
            probs = self.actor(state_t)
            log_prob = torch.log(probs[0][action])
            actor_loss = -log_prob * advantage

            # Critic loss
            critic_loss = (target - value) ** 2

            self.optimizerA.zero_grad()
            actor_loss.backward(retain_graph=True)
            self.optimizerA.step()

            self.optimizerC.zero_grad()
            critic_loss.backward()
            self.optimizerC.step()
        self.memory = []
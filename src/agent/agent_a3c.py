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
        self.actor = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Linear(8, action_size)
        )
        self.critic = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

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
        pass  # A3C는 step마다 별도 학습 없음

    def train(self, gamma=0.99):
        for state, action, reward, next_state, done in self.memory:
            state_t = torch.FloatTensor(state)
            next_state_t = torch.FloatTensor(next_state)
            probs, value = self.model(state_t)
            _, next_value = self.model(next_state_t)
            target = reward + (0 if done else gamma * next_value.item())
            advantage = target - value.item()

            prob_action = probs[0][action]
            log_prob = torch.log(prob_action + 1e-8)  # log(0) 방지
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

        probs_np = probs.detach().cpu().numpy().flatten()

        # 안전성 검사 및 정규화
        if not np.all(np.isfinite(probs_np)):
            print("[WARN] NaN/Inf in probs → fallback to uniform")
            probs_np = np.ones(self.action_size) / self.action_size
        else:
            probs_np = np.clip(probs_np, 0, 1)
            total = probs_np.sum()
            if total == 0 or not np.isfinite(total):
                print("[WARN] Sum of probs invalid → fallback to uniform")
                probs_np = np.ones(self.action_size) / self.action_size
            else:
                probs_np = probs_np / total

        if not np.isclose(probs_np.sum(), 1.0, atol=1e-3):
            print("[WARN] Probs not normalized → fallback to uniform")
            probs_np = np.ones(self.action_size) / self.action_size

        return np.random.choice(self.action_size, p=probs_np)

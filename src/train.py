# train.py
# End-to-end training / evaluation script with risk-aware reward options

import argparse
import os
from tqdm import tqdm
import torch
import numpy as np

from functions import getStockDataVec, getState, formatPrice
from visualizer import plot_trades

# ---------- CLI ---------- #
parser = argparse.ArgumentParser()
parser.add_argument('--stock', type=str, default='^GSPC')
parser.add_argument('--window', type=int, default=30)
parser.add_argument('--episodes', type=int, default=100)
parser.add_argument('--agent', type=str, default='dqn',
                    choices=['dqn', 'pg', 'ac', 'a2c', 'a3c', 'deepsarsa'])
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--cash', type=float, default=10)
parser.add_argument('--normalizer', type=float, default=1)
parser.add_argument('--indicator', type=str, default='none',
                    choices=['none', 'sharpe', 'kelly', 'var', 'cvar'])
parser.add_argument('--lambda_sharpe', type=float, default=0.1)
parser.add_argument('--sharpe_window', type=int, default=20)
parser.add_argument('--lambda_kelly', type=float, default=0.1)
parser.add_argument('--kelly_window', type=int, default=20)
parser.add_argument('--alpha_var', type=float, default=0.05)
parser.add_argument('--lambda_var', type=float, default=0.1)
parser.add_argument('--alpha_cvar', type=float, default=0.05)
parser.add_argument('--tau_cvar', type=float, default=-0.05)
parser.add_argument('--beta_cvar', type=float, default=0.1)

args = parser.parse_args()

# ---------- dynamic agent import ---------- #
if args.agent == 'dqn':
    from agent.agent_dqn import AgentDQN as _Agent
elif args.agent == 'pg':
    from agent.agent_pg import AgentPG as _Agent
elif args.agent == 'ac':
    from agent.agent_ac import AgentAC as _Agent
elif args.agent == 'a2c':
    from agent.agent_a2c import AgentA2C as _Agent
elif args.agent == 'a3c':
    from agent.agent_a3c import AgentA3C as _Agent
elif args.agent == 'deepsarsa':
    from agent.agent_deepsarsa import AgentDeepSARSA as _Agent
else:
    raise ValueError("Unknown agent type")

agent = _Agent(args.window, lr=args.lr)

# ---------- constants ---------- #
EPS = 1e-8
SCALE_S = 5.0
PEN_MAX = 5.0
MIN_N = 5
MIN_N_RISK = 20

# ---------- safe metric helpers ---------- #
def safe_sharpe(returns):
    if len(returns) < MIN_N:
        return 0.0
    mu, sigma = np.mean(returns), np.std(returns)
    if sigma < EPS:
        return 0.0
    s = mu / sigma
    return float(np.tanh(s / SCALE_S)) if np.isfinite(s) else 0.0

def safe_kelly(returns):
    if len(returns) < MIN_N:
        return 0.0
    wins   = [r for r in returns if r > 0]
    losses = [r for r in returns if r <= 0]
    if not losses:
        return 0.0
    wins_mean = np.mean(wins) if wins else EPS
    losses_mean = abs(np.mean(losses)) + EPS
    p_hat = (len(wins) + 1) / (len(returns) + 2)
    ratio = wins_mean / losses_mean
    k = p_hat - (1 - p_hat) / ratio
    if not np.isfinite(k):
        k = 0.0
    return float(np.clip(k, -1.0, 1.0))

def safe_var(returns, alpha, tau):
    if len(returns) < MIN_N_RISK:
        return 0.0
    var = np.quantile(returns, alpha)
    pen = max(0.0, -var - tau)
    return float(np.clip(pen, 0, PEN_MAX))

def safe_cvar(returns, alpha, tau):
    if len(returns) < MIN_N_RISK:
        return 0.0
    var_thr = np.quantile(returns, alpha)
    tail = [r for r in returns if r <= var_thr]
    if not tail:
        return 0.0
    cvar = np.mean(tail)
    pen_raw = max(0.0, tau - cvar)
    pen = np.log1p(np.exp(pen_raw))
    return float(np.clip(pen, 0, PEN_MAX))

# ---------- data loading ---------- #
data_raw = getStockDataVec(args.stock)
normalizer = data_raw[0] if args.normalizer == 1 else args.normalizer
data_raw = [p / normalizer for p in data_raw]

split_idx = int(len(data_raw) * 0.8)
train_prices = data_raw[:split_idx]
eval_prices = data_raw[split_idx:]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("PyTorch device:", device)

batch_size = 32
l_train = len(train_prices) - 1
max_window = max(args.sharpe_window, args.kelly_window, MIN_N_RISK)

# ---------- training loop ---------- #
for e in range(args.episodes + 1):
    state = getState(train_prices, 0, args.window + 1)
    total_profit, cash = 0.0, args.cash
    agent.inventory = []
    trade_returns = []
    prev_metric = 0.0

    pbar = tqdm(range(l_train), desc=f"Episode {e}/{args.episodes}", unit='step')
    for t in pbar:
        action = agent.act(state)
        next_state = getState(train_prices, t + 1, args.window + 1)
        reward = 0.0

        if action == 1 and cash >= train_prices[t]:
            agent.inventory.append(train_prices[t])
            cash -= train_prices[t]
            avg_price = sum(agent.inventory) / len(agent.inventory)
            agent.inventory = [avg_price] * len(agent.inventory)

        elif action == 2 and agent.inventory:
            buy_price = agent.inventory.pop(0)
            trade_profit = train_prices[t] - buy_price
            reward = trade_profit
            total_profit += trade_profit
            cash += train_prices[t]

            trade_return = trade_profit / (buy_price + EPS)
            trade_returns.append(trade_return)
            if len(trade_returns) > max_window:
                trade_returns.pop(0)

            if args.indicator == 'sharpe':
                cur = safe_sharpe(trade_returns[-args.sharpe_window:])
                delta = cur - prev_metric
                reward += args.lambda_sharpe * delta
                prev_metric = cur

            elif args.indicator == 'kelly':
                cur = safe_kelly(trade_returns[-args.kelly_window:])
                delta = cur - prev_metric
                if not np.isfinite(delta):
                    delta = 0.0
                reward += args.lambda_kelly * delta
                prev_metric = cur

            elif args.indicator == 'var':
                penalty = safe_var(trade_returns, args.alpha_var, tau=0.0)
                reward -= args.lambda_var * penalty

            elif args.indicator == 'cvar':
                penalty = safe_cvar(trade_returns, args.alpha_cvar, args.tau_cvar)
                reward -= args.beta_cvar * penalty

        if t == l_train - 1 and agent.inventory:
            for buy_price in agent.inventory:
                trade_profit = train_prices[t] - buy_price
                reward += trade_profit
                total_profit += trade_profit
                cash += train_prices[t]
                trade_returns.append(trade_profit / (buy_price + EPS))
            agent.inventory = []

        reward = float(np.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0))
        agent.remember(state, action, reward, next_state, t == l_train - 1)
        agent.train_step(batch_size)
        state = next_state

        pbar.set_postfix({"Profit": formatPrice(total_profit * normalizer),
                          "Cash": formatPrice(cash * normalizer)})

        if t == l_train - 1:
            agent.train()

    if e % 10 == 0:
        os.makedirs('models', exist_ok=True)
        if args.agent == 'ac':
            torch.save(agent.actor.state_dict(), f"models/{args.agent}_actor_ep{e}.pt")
            torch.save(agent.critic.state_dict(), f"models/{args.agent}_critic_ep{e}.pt")
        else:
            torch.save(agent.model.state_dict(), f"models/{args.agent}_ep{e}.pt")

# ---------- evaluation ---------- #
print("Starting evaluation…")
agent.is_eval = True
l_eval = len(eval_prices) - 1
state = getState(eval_prices, 0, args.window + 1)

total_profit, cash = 0.0, args.cash
buy_steps, sell_steps, portfolio_values = [], [], []

log_dir = os.path.join("logs", args.agent)
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"{args.stock}_{args.indicator}.txt")

with open(log_file, 'w') as f:
    f.write(f"Evaluation of {args.stock} with {args.agent} agent and {args.indicator} indicator\n")
    f.write(str(args) + "\n")

    for t in range(l_eval):
        action = agent.act(state)
        next_state = getState(eval_prices, t + 1, args.window + 1)

        if action == 1 and cash >= eval_prices[t]:
            agent.inventory.append(eval_prices[t])
            cash -= eval_prices[t]
            avg_price = sum(agent.inventory) / len(agent.inventory)
            agent.inventory = [avg_price] * len(agent.inventory)
            buy_steps.append(t)
            f.write(f"DAY {t} | PRICE {formatPrice(eval_prices[t] * normalizer)} | ACTION BUY\n")

        elif action == 2 and agent.inventory:
            buy_price = agent.inventory.pop(0)
            profit = eval_prices[t] - buy_price
            total_profit += profit
            cash += eval_prices[t]
            sell_steps.append(t)
            f.write(f"DAY {t} | PRICE {formatPrice(eval_prices[t] * normalizer)} | ACTION SELL\n")

        portfolio_values.append(cash + len(agent.inventory) * eval_prices[t])
        state = next_state

    if agent.inventory:
        last_price = eval_prices[-1]
        for buy_price in agent.inventory:
            total_profit += last_price - buy_price
            cash += last_price
            sell_steps.append(l_eval - 1)
            f.write(f"DAY {l_eval - 1} | PRICE {formatPrice(last_price * normalizer)} | ACTION LIQUIDATE\n")
        agent.inventory = []

    f.write(f"Initial Cash: {formatPrice(args.cash * normalizer)}\n")
    f.write(f"Total Profit: {formatPrice(total_profit * normalizer)}\n")
    f.write(f"Final Cash:  {formatPrice(cash * normalizer)}\n")
    f.write(f"Price Change: {formatPrice(eval_prices[0] * normalizer)} → {formatPrice(eval_prices[-1] * normalizer)} "
            f"({(eval_prices[-1] - eval_prices[0]) / eval_prices[0] * 100:.2f}%)\n")

# ---------- plot ---------- #
plot_dir = os.path.join("plots", args.agent)
os.makedirs(plot_dir, exist_ok=True)

initial_cash = args.cash * normalizer
final_cash = cash * normalizer
roi = (final_cash - initial_cash) / initial_cash * 100 if initial_cash != 0 else 0

plot_trades(
    [p * normalizer for p in eval_prices],
    buy_steps,
    sell_steps,
    [v * normalizer for v in portfolio_values],
    title=(f"{args.stock} Trading Result ({args.agent}, {args.indicator})\n"
           f"{formatPrice(initial_cash)} → {formatPrice(final_cash)} ({roi:.2f}%) "
           f"while price {formatPrice(eval_prices[0])} → {formatPrice(eval_prices[-1])} "
           f"({(eval_prices[-1] - eval_prices[0]) / eval_prices[0] * 100:.2f}%)"),
    save_path=os.path.join(plot_dir, f"{args.stock}_{args.indicator}.png")
)

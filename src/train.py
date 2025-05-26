import argparse
from functions import *
import torch
from tqdm import tqdm
import os
from visualizer import plot_trades

parser = argparse.ArgumentParser()
parser.add_argument('--stock', type=str, default='^GSPC')
parser.add_argument('--window', type=int, default=10, help="Number of previous days to consider for the state")
parser.add_argument('--episodes', type=int, default=100, help="Number of training episodes")
parser.add_argument('--agent', type=str, default='dqn', choices=['dqn', 'pg', 'ac', 'a2c', 'a3c', 'deepsarsa'])
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--cash', type=float, default=1000, help="Initial cash amount for trading")
parser.add_argument('--normalizer', type=float, default=1, help="Normalization factor for stock prices")
args = parser.parse_args()

if args.agent == 'dqn':
    from agent.agent_dqn import AgentDQN
    agent = AgentDQN(args.window)
elif args.agent == 'pg':
    from agent.agent_pg import AgentPG
    agent = AgentPG(args.window, lr=args.lr)
elif args.agent == 'ac':
    from agent.agent_ac import AgentAC
    agent = AgentAC(args.window, lr=args.lr)
elif args.agent == 'a2c':
    from agent.agent_a2c import AgentA2C
    agent = AgentA2C(args.window, lr=args.lr)
elif args.agent == 'a3c':
    from agent.agent_a3c import AgentA3C
    agent = AgentA3C(args.window, lr=args.lr)
elif args.agent == 'deepsarsa':
    from agent.agent_deepsarsa import AgentDeepSARSA
    agent = AgentDeepSARSA(args.window, lr=args.lr)
else:
    raise ValueError("Unknown agent type. Please choose from 'dqn', 'pg', 'ac', 'a2c', 'a3c', or 'deepsarsa'.")

data = getStockDataVec(args.stock, normalizer=args.normalizer)
data = data[:100] if len(data) > 100 else data

l = len(data) - 1
batch_size = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("PyTorch device:", device)

for e in range(args.episodes + 1):
    state = getState(data, 0, args.window + 1)
    total_profit = 0
    agent.inventory = []
    cash = args.cash  # 초기 보유 금액

    pbar = tqdm(range(l), desc="Training Progress", unit="step")
    pbar.set_description(f"Episode {e}/{args.episodes}")
    for t in pbar:
        action = agent.act(state)
        next_state = getState(data, t + 1, args.window + 1)
        reward = 0

        if action == 1:
            # 구매 시 보유 금액 확인
            if cash >= data[t]:
                agent.inventory.append(data[t])
                cash -= data[t]
                # average every buy price
                avg_price = sum(agent.inventory) / len(agent.inventory)
                agent.inventory = [avg_price] * len(agent.inventory)
            # else: 구매 불가, 아무 일도 하지 않음
        elif action == 2 and len(agent.inventory) > 0:
            bought_price = agent.inventory.pop(0)
            reward = data[t] - bought_price
            total_profit += data[t] - bought_price
            cash += data[t]

        done = True if t == l - 1 else False

        # 마지막 step에서 남은 주식 모두 강제 매도
        if done and len(agent.inventory) > 0:
            for price in agent.inventory:
                reward += data[t] - price
                total_profit += data[t] - price
                cash += data[t]
            agent.inventory = []

        agent.remember(state, action, reward, next_state, done)
        agent.train_step(batch_size)

        state = next_state

        pbar.set_postfix({"Profit": formatPrice(total_profit), "Cash": formatPrice(cash)})

        if done:
            agent.train()

    if e % 10 == 0:
        os.makedirs("models", exist_ok=True)
        if args.agent == 'ac':
            torch.save(agent.actor.state_dict(), f"models/model_{args.agent}_actor_ep{e}.pt")
            torch.save(agent.critic.state_dict(), f"models/model_{args.agent}_critic_ep{e}.pt")
        else:
            torch.save(agent.model.state_dict(), f"models/model_{args.agent}_ep{e}.pt")

# Evaluation
data = getStockDataVec(args.stock, normalizer=args.normalizer)
data = data[100:200] if len(data) > 200 else data

agent.is_eval = True
state = getState(data, 0, args.window + 1)
total_profit = 0
agent.inventory = []
cash = args.cash
l = len(data) - 1
print("Starting evaluation...")

buy_steps = []
sell_steps = []
portfolio_values = []

for t in range(l):
    action = agent.act(state)
    next_state = getState(data, t + 1, args.window + 1)
    reward = 0

    if action == 1:
        if cash >= data[t]:
            agent.inventory.append(data[t])
            cash -= data[t]
            print(f"[{t:3d}] Buy: " + formatPrice(data[t]))
            avg_price = sum(agent.inventory) / len(agent.inventory)
            agent.inventory = [avg_price] * len(agent.inventory)
            buy_steps.append(t)
        # else: 구매 불가, 아무 일도 하지 않음
    elif action == 2 and len(agent.inventory) > 0:
        bought_price = agent.inventory.pop(0)
        reward = max(data[t] - bought_price, 0)
        total_profit += data[t] - bought_price
        cash += data[t]
        print(f"[{t:3d}] Sell: " + formatPrice(data[t]) + " | Profit: " + formatPrice(data[t] - bought_price))
        sell_steps.append(t)

    # 포트폴리오 가치 계산 (현금 + 보유 주식 수 * 현재가)
    portfolio_value = cash + len(agent.inventory) * data[t]
    portfolio_values.append(portfolio_value)

    done = True if t == l - 1 else False
    if done:
        print(f"[{t:3d}] End of episode. Total Profit: " + formatPrice(total_profit))
        if len(agent.inventory) > 0:
            for price in agent.inventory:
                total_profit += data[t] - price
                cash += data[t]
                print(f"[{t:3d}] Sell remaining: " + formatPrice(data[t]) + " | Profit: " + formatPrice(data[t] - price))
                sell_steps.append(t)
            agent.inventory = []

    state = next_state

print(f"Total Profit: {formatPrice(total_profit)}")
print(f"Final Cash: {formatPrice(cash)}")

# Plot and save the trading result
os.makedirs("plots", exist_ok=True)
plot_trades(
    data,
    buy_steps,
    sell_steps,
    portfolio_values=portfolio_values,
    title=f"{args.stock} Trading Result ({args.agent}, {formatPrice(total_profit)}/{args.cash})",
    save_path=f"plots/{args.stock}_{args.agent}_trading_result.png"
)

import argparse
from functions import *
import torch
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--stock', type=str, default='^GSPC')
parser.add_argument('--window', type=int, default=10, help="Number of previous days to consider for the state")
parser.add_argument('--episodes', type=int, default=1000, help="Number of training episodes")
parser.add_argument('--agent', type=str, default='dqn', choices=['dqn', 'pg', 'ac', 'a2c', 'a3c', 'deepsarsa'])
parser.add_argument('--lr', type=float, default=0.001)
args = parser.parse_args()

if args.agent == 'dqn':
    from agent.agent import Agent
    agent = Agent(args.window)
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

data = getStockDataVec(args.stock)
data = data[:30] if len(data) > 30 else data

l = len(data) - 1
batch_size = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("PyTorch device:", device)

for e in range(args.episodes + 1):
    state = getState(data, 0, args.window + 1)
    total_profit = 0
    agent.inventory = []

    pbar = tqdm(range(l), desc="Training Progress", unit="step")
    pbar.set_description(f"Episode {e}/{args.episodes} - Profit: {formatPrice(total_profit)}")
    for t in pbar:
        action = agent.act(state)
        next_state = getState(data, t + 1, args.window + 1)
        reward = 0

        if action == 1:
            agent.inventory.append(data[t])
        elif action == 2 and len(agent.inventory) > 0:
            bought_price = agent.inventory.pop(0)
            reward = max(data[t] - bought_price, 0)
            total_profit += data[t] - bought_price

        pbar.set_postfix({"Profit": formatPrice(total_profit)})

        done = True if t == l - 1 else False
        # Different agents have different memory and training methods
        if args.agent == 'dqn':
            agent.memory.append((state, action, reward, next_state, done))
        elif args.agent == 'pg':
            agent.remember(state, action, reward)
        elif args.agent == 'ac':
            agent.remember(state, action, reward, next_state, done)
        elif args.agent == 'a2c':
            agent.remember(state, action, reward, next_state, done)
        elif args.agent == 'a3c':
            agent.remember(state, action, reward, next_state, done)
        elif args.agent == 'deepsarsa':
            agent.memory.append((state, action, reward, next_state, done))

        state = next_state

        if done:
            if args.agent == 'pg':
                agent.train()
            elif args.agent == 'ac':
                agent.train()
            elif args.agent == 'a2c':
                agent.train()
            elif args.agent == 'a3c':
                agent.train()
            elif args.agent == 'deepsarsa':
                agent.train()

        if args.agent == 'dqn' and len(agent.memory) > batch_size:
            agent.expReplay(batch_size)
        elif args.agent == 'deepsarsa' and len(agent.memory) > batch_size:
            agent.expReplay(batch_size)

    if e % 10 == 0:
        torch.save(agent.model.state_dict(), f"models/model_{args.agent}_ep{e}.pt")

import argparse
import torch
from functions import *

parser = argparse.ArgumentParser()
parser.add_argument('--stock', type=str, required=True)
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--agent', type=str, required=True, choices=['dqn', 'pg', 'ac', 'a2c', 'a3c', 'deepsarsa'])
parser.add_argument('--window', type=int, default=10)
args = parser.parse_args()

if args.agent == 'dqn':
    from agent.agent import Agent
    agent = Agent(args.window, is_eval=True, model_name=args.model)
elif args.agent == 'pg':
    from agent.agent_pg import AgentPG
    agent = AgentPG(args.window)
    agent.model.load_state_dict(torch.load("models/" + args.model, map_location=torch.device("cpu")))
elif args.agent == 'ac':
    from agent.agent_ac import AgentAC
    agent = AgentAC(args.window)
    agent.actor.load_state_dict(torch.load("models/" + args.model, map_location=torch.device("cpu")))
elif args.agent == 'a2c':
    from agent.agent_a2c import AgentA2C
    agent = AgentA2C(args.window)
    agent.model.load_state_dict(torch.load("models/" + args.model, map_location=torch.device("cpu")))
elif args.agent == 'a3c':
    from agent.agent_a3c import AgentA3C
    agent = AgentA3C(args.window)
    agent.model.load_state_dict(torch.load("models/" + args.model, map_location=torch.device("cpu")))
elif args.agent == 'deepsarsa':
    from agent.agent_deepsarsa import AgentDeepSARSA
    agent = AgentDeepSARSA(args.window)
    agent.model.load_state_dict(torch.load("models/" + args.model, map_location=torch.device("cpu")))
else:
    raise ValueError("Unknown agent type.")

data = getStockDataVec(args.stock)
l = len(data) - 1

state = getState(data, 0, args.window + 1)
total_profit = 0
agent.inventory = []

for t in range(l):
    action = agent.act(state)
    next_state = getState(data, t + 1, args.window + 1)
    reward = 0

    if action == 1:
        agent.inventory.append(data[t])
        print("Buy: " + formatPrice(data[t]))
    elif action == 2 and len(agent.inventory) > 0:
        bought_price = agent.inventory.pop(0)
        reward = max(data[t] - bought_price, 0)
        total_profit += data[t] - bought_price
        print("Sell: " + formatPrice(data[t]) + " | Profit: " + formatPrice(data[t] - bought_price))

    done = True if t == l - 1 else False
    if args.agent in ['dqn', 'deepsarsa']:
        agent.memory.append((state, action, reward, next_state, done))
    elif args.agent == 'pg':
        agent.remember(state, action, reward)
    elif args.agent in ['ac', 'a2c', 'a3c']:
        agent.remember(state, action, reward, next_state, done)

    state = next_state

    if done:
        print("--------------------------------")
        print(args.stock + " Total Profit: " + formatPrice(total_profit))
        print("--------------------------------")

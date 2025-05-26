#!/bin/bash

AGENTS=("dqn" "pg" "ac" "a2c" "a3c" "deepsarsa")
STOCKS=("hangsen_ours" "japan_ni_ours" "kosdaq_ours" "kospi_ours" "nasdaq_ours")
CASH=(500 500 200 200 200)
NORMALIZER=(100 100 10 10 40)
EPISODES=10

for agent in "${AGENTS[@]}"; do
  for i in "${!STOCKS[@]}"; do
    stock="${STOCKS[$i]}"
    cash="${CASH[$i]}"
    normalizer="${NORMALIZER[$i]}"
    echo "Running: python train.py --agent=$agent --episodes=$EPISODES --stock=$stock --cash=$cash --normalizer=$normalizer"
    python train.py --agent="$agent" --episodes="$EPISODES" --stock="$stock" --cash="$cash" --normalizer="$normalizer"
  done
done

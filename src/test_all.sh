#!/bin/bash

AGENTS=("pg" "dqn" "ac" "a2c" "a3c" "deepsarsa")
BASE_STOCKS=("hangsen_ours" "japan_ni_ours" "kosdaq_ours" "kospi_ours" "nasdaq_ours")
INDICATORS=("none" "sharpe" "kelly" "var" "cvar")
EPISODES=10
KFOLDS=5

for agent in "${AGENTS[@]}"; do
  for stock in "${BASE_STOCKS[@]}"; do
    for fold in $(seq 1 $KFOLDS); do
      stock_fold="${stock}_fold${fold}"
      for indicator in "${INDICATORS[@]}"; do
        echo "Running: python train.py --agent=$agent --episodes=$EPISODES --stock=$stock_fold --indicator=$indicator"
        python train.py --agent="$agent" --episodes="$EPISODES" --stock="$stock_fold" --indicator="$indicator"
      done
    done
  done
done

# LunarLander Reinforcement Learning

This is a learning project about neural networks and reinforcement learning. We are using Gymnasium's `LunarLander-v3` environment and building understanding step by step instead of treating an agent as a finished solution.

## Project plan

1. Explore the environment, its observations, actions, and rewards.
2. Train an agent in discrete mode, where it chooses among four actions.
3. Measure how often it lands safely across many different seeds.
4. Later, explore continuous mode, where the agent controls engine strength.

The first exploration compares doing nothing with manually controlled actions. It helps us see how actions affect the lander's movement, but it is not a trained agent. See [explorations/01_actions/compare_actions.md](explorations/01_actions/compare_actions.md).

## Reward idea

The environment already provides a reward. We will use it as our baseline and only consider changes if experiments reveal a specific problem. See [explorations/02_reward/reward_breakdown.md](explorations/02_reward/reward_breakdown.md) for a short explanation, and run [reward_simulation.py](explorations/02_reward/reward_simulation.py) to compare no-action, random, and manual control.

## Current status

- Gymnasium's LunarLander-v3 is installed in the local `.venv/` environment.
- Exploration 1: manual action comparison in `explorations/01_actions/`.
- Exploration 2: default reward notes and live simulation in `explorations/02_reward/`.
- Exploration 3: a small deterministic Q-table example in `explorations/03_q_table/`.
- Exploration 4: an interactive Q-learning sandbox in `explorations/04_q_learning/`.
- No neural network or training algorithm has been added yet.

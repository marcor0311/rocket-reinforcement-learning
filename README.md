# LunarLander Reinforcement Learning

This is a learning project about neural networks and reinforcement learning. We are using Gymnasium's `LunarLander-v3` environment and building understanding step by step instead of treating an agent as a finished solution.

## Project plan

1. Explore the environment, its observations, actions, and rewards.
2. Train an agent in discrete mode, where it chooses among four actions.
3. Measure how often it lands safely across many different seeds.
4. Later, explore continuous mode, where the agent controls engine strength.

The first experiment compares doing nothing with manually controlled actions. It helps us see how actions affect the lander's movement, but it is not a trained agent. See [compare_actions.md](compare_actions.md).

## Reward idea

The environment already provides a reward. Our early idea is to understand that reward first, then consider whether to adjust it to encourage a safe, upright, slower landing and discourage unnecessary engine use or long episodes. Any changes will be experiments to evaluate, not assumptions that the agent must follow.

## Current status

- Gymnasium's LunarLander-v3 is installed in the local `.venv/` environment.
- The side-by-side manual comparison demo is in `compare_actions.py`.
- No neural network or training algorithm has been added yet.

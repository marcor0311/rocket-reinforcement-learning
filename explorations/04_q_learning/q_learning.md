# Exploration 4: See Q-learning respond to LunarLander

## Goal

Take an action and see how the reward changes its value in the Q-table. The window focuses on the lander, four action values for its current kind of situation, and the value that changed after your action.

## Controls

- `Left` / `Right`: side engines
- `Up` or `Space`: main engine
- No key: do nothing
- `R`: restart (the table remembers experience until you quit)
- `Q` or `Esc`: quit

Each press advances one step. Try repeating an action or restarting and returning to a similar situation. Watch the reward and the selected action's value.

## Run it

From the project root:

```bash
.venv/bin/python explorations/04_q_learning/q_learning.py
```

This uses real LunarLander rewards and a small Q-table. It is not a neural network, and it does not choose actions for you yet: you control the lander while the table records experience. The table groups three observations into ranges to keep the example small, so it is only a learning aid.
# Exploration 5: Let the Q-table choose

## Goal

See how an agent uses its Q-table to choose a LunarLander action. The agent usually picks the action with the highest value for the current situation. Sometimes it picks a random action to explore. This is called **epsilon-greedy**.

## Controls

- The agent starts automatically; press `A` to pause or resume it
- `Left` / `Right`: manually fire the side engines
- `Up` or `Space`: manually fire the main engine
- `R`: restart the episode and start the agent again (keeps the table for this session)
- `Q` or `Esc`: quit

With the agent on, it takes one step every quarter second. The status text says whether it used the table's best value or explored randomly. The screen also shows the chosen action, Gymnasium reward, and how that action's value changed. Press an arrow or engine key to switch back to manual control.

## What to notice

The table begins with values at zero. When several actions tie for the best value, the agent chooses among those tied actions at random. As it gets rewards, the values change; the agent then favors actions with higher values. With epsilon `0.2`, about 20% of its choices are random exploration, while the other choices use the table. Because the state table is very small, this is only a way to observe the choice rule, not an effective landing agent.

## Run it

From the project root:

```bash
.venv/bin/python explorations/05_q_learning_agent/q_learning_agent.py
```

This is still tabular Q-learning, not a neural network. You can compare letting the agent choose with taking manual control yourself.

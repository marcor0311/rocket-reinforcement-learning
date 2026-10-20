# LunarLander v3

## Discrete mode: observations and actions

The agent receives 8 values from the environment at each step:

| # | Observation | What it means |
|---|---|---|
| 1 | Horizontal position | Where the lander is from left to right |
| 2 | Vertical position | How high the lander is above the ground |
| 3 | Horizontal speed | How fast the lander moves left or right |
| 4 | Vertical speed | How fast the lander moves up or down |
| 5 | Angle | How much the lander is tilted |
| 6 | Rotation speed | How fast the lander is turning |
| 7 | Left leg contact | Whether the left leg touches the ground (0 or 1) |
| 8 | Right leg contact | Whether the right leg touches the ground (0 or 1) |

In discrete mode, the agent chooses one of 4 actions at each step:

| Action | What it does |
|---|---|
| 0 | Do nothing |
| 1 | Fire the left side engine |
| 2 | Fire the main engine |
| 3 | Fire the right side engine |

See [compare_actions.md](compare_actions.md) for the side-by-side manual action experiment.

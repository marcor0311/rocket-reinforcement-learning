# Compare Actions

## Purpose

This experiment compares a lander that always does nothing with a lander controlled by a person. Both start from the same seed so their starting situations are as similar as possible.

## Observations

At each step, Gymnasium gives the agent 8 values:

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

## Discrete actions

The agent chooses one of four actions at each step:

| Action | What it does |
|---|---|
| 0 | Do nothing |
| 1 | Fire the left side engine |
| 2 | Fire the main engine |
| 3 | Fire the right side engine |

The side engines rotate the lander. Once tilted, the lander can move horizontally. The main engine pushes upward.

## Controls

| Key | Discrete action | Effect |
|---|---:|---|
| No key | 0 | Do nothing |
| Left Arrow | 1 | Fire the left side engine |
| Up Arrow or Space | 2 | Fire the main engine |
| Right Arrow | 3 | Fire the right side engine |

## What to observe

Compare the two landers' vertical speed, score, and landing outcome. Notice that an action's effect depends on the lander's current state. Firing the main engine continuously can push the lander upward, while doing nothing can sometimes be enough when the starting motion is favorable.

This is a manual demonstration, not a trained agent. One seed or a few trials cannot tell us whether a policy will land safely across many different starting conditions. We need repeated evaluation across many seeds for that.

## Run the demo

```bash
python compare_actions.py
```

The left lander does nothing. Control the right lander with the keys above. Press **R** to restart both with the same seed or **Q** to quit.

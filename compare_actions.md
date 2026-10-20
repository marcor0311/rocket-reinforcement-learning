# Compare Actions

## Purpose

This experiment compares a lander that always does nothing with a lander controlled by a person. Both start from the same seed so their starting situations are as similar as possible.

## Controls

| Key | Discrete action | Effect |
|---|---:|---|
| No key | 0 | Do nothing |
| Left Arrow | 1 | Fire the left side engine |
| Up Arrow or Space | 2 | Fire the main engine |
| Right Arrow | 3 | Fire the right side engine |

The side engines rotate the lander. Once tilted, the lander can move horizontally. The main engine pushes upward.

## What to observe

Compare the two landers' vertical speed, score, and landing outcome. Notice that an action's effect depends on the lander's current state. Firing the main engine continuously can push the lander upward, while doing nothing can sometimes be enough when the starting motion is favorable.

This is a manual demonstration, not a trained agent. One seed or a few trials cannot tell us whether a policy will land safely across many different starting conditions. We need repeated evaluation across many seeds for that.

## Run the demo

```bash
python compare_actions.py
```

The left lander does nothing. Control the right lander with the keys above. Press **R** to restart both with the same seed or **Q** to quit.

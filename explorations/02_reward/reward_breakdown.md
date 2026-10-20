# LunarLander's Default Reward

Gymnasium already gives a reward after every action. The agent's **episode score** is the sum of those rewards.
This reward is feedback from the environment; the agent still needs a separate training process to learn what to do.

## What affects each step

Gymnasium builds a shaping score from four signals:

- Closer to the landing pad: better.
- Slower movement: better.
- More upright: better.
- Each leg touching the ground: adds a positive signal.

The step reward uses the **change** in that shaping score. Improving the signals earns reward; getting worse loses reward. It also charges for engine use:

| Engine use | Reward cost per step |
|---|---:|
| Main engine | −0.30 |
| Side engine | −0.03 |

There is no fuel tank to empty. The engine cost encourages efficient use. There is also no separate penalty just for taking another step.

## How an episode ends

- Safe landing: **+100** terminal reward.
- Crash or moving beyond the horizontal bounds: **−100** terminal reward.

These terminal rewards replace the usual reward for the final step.

## Why keep this reward for now?

It already represents most of our goals: approach the pad, slow down, stay upright, and land. We will use it as our baseline, then consider a custom reward only if experiments reveal a specific problem.

Run [reward_simulation.py](reward_simulation.py) to compare three landers from the same starting seed: one does nothing, one picks random actions, and you control the third. Each panel shows its score and reward components. The leader is highlighted, and the final view compares scores and landing outcomes.

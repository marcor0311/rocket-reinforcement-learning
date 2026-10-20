# Exploration 3: How a Q-table represents decisions

A Q-table stores one value for each **state and action** pair. A state is the situation the lander is in; an action is one of the four discrete controls. After learning, the agent can choose the action with the highest value for its current state.

LunarLander observations are continuous numbers, so a plain table cannot use every possible number as a separate key. This demo groups three selected observations—horizontal position, vertical speed, and tilt—into ranges called **bins**. The resulting integers form a compact state key.

## Run it

From the project root:

```bash
python explorations/03_q_table/q_table_demo.py
```

The example initializes every action value to zero. It does not interact with Gymnasium, update values, or train an agent. It only demonstrates the table's shape and the effect of binning. Here, 3 observations with 3 bins each create 27 states; four actions give 108 values. Using all 8 observations with 3 bins each would create 6,561 states and 26,244 values.

## Is it deterministic?

This demo is deterministic: the same readings always produce the same bins and output. A fixed Gymnasium seed and fixed actions also make an episode repeatable in this environment. A learning run can add randomness by choosing exploratory actions (for example, randomly trying an action instead of choosing the current best one), and changing the seed changes the starting conditions. Deterministic updates do not mean every training run or policy must be deterministic.

## Conclusion

A Q-table is useful for learning how an agent can connect states, actions, and values. For LunarLander, though, it is likely to be inefficient: even with only 3 bins per observation, all 8 observations create 6,561 states and 26,244 state-action values. Binning also loses detail from the original observations. A neural network can estimate action values without storing a separate entry for every binned combination, which is why DQN is a better fit to explore after we understand the Q-learning idea.

## What comes next?

Next, we can add the Q-learning update rule and let the table learn from transitions and Gymnasium rewards. Binning makes a table easy to understand, but it loses precision and grows quickly as we add observations or bins. That limitation motivates neural networks such as DQN later.

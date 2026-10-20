"""Show how binned observations map to action values in a tiny Q-table."""

from itertools import product


OBSERVATION_NAMES = ("horizontal position", "vertical speed", "tilt")
# Each boundary divides one continuous observation into a few readable ranges.
BINS = (
    (-0.25, 0.25),       # x: left / near center / right
    (-0.2, 0.2),         # vertical speed: rising / slow / falling
    (-0.15, 0.15),       # angle: tilted left / nearly upright / tilted right
)
ACTION_NAMES = ("do nothing", "left", "main", "right")


def bin_value(value: float, boundaries: tuple[float, ...]) -> int:
    """Return which interval contains a value; intervals are numbered from 0."""
    for index, boundary in enumerate(boundaries):
        if value < boundary:
            return index
    return len(boundaries)


def discretize(observation: tuple[float, ...]) -> tuple[int, ...]:
    """Convert selected continuous readings into an integer state key."""
    return tuple(bin_value(value, cuts) for value, cuts in zip(observation, BINS))


def all_states() -> list[tuple[int, ...]]:
    """Enumerate all possible binned keys for this deliberately tiny example."""
    sizes = tuple(len(cuts) + 1 for cuts in BINS)
    return list(product(*(range(size) for size in sizes)))


def main() -> None:
    states = all_states()
    # A Q-table starts with a value for every state/action pair. These zeroes
    # are placeholders; a learning process would update them from rewards.
    q_table = {state: [0.0] * len(ACTION_NAMES) for state in states}

    readings = (-0.4, 0.05, 0.2)
    state = discretize(readings)
    print("A small Q-table (not a trained agent)")
    print("Continuous readings:")
    for name, value in zip(OBSERVATION_NAMES, readings):
        print(f"  {name}: {value:+.2f}")
    print(f"Binned state key: {state}")
    print("Action values in this state:")
    for action, value in zip(ACTION_NAMES, q_table[state]):
        print(f"  {action:12} {value:+.1f}")
    print("Chosen action: do nothing (all values tie; this demo breaks ties by order)")
    print()
    print("Table size for these 3 selected observations:")
    print(f"  {len(states)} binned states x {len(ACTION_NAMES)} actions = "
          f"{len(states) * len(ACTION_NAMES)} values")
    print("With all 8 LunarLander observations, even 3 bins each would mean "
          "3^8 = 6,561 states, or 26,244 action values.")


if __name__ == "__main__":
    main()

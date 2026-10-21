"""Interactive visual sandbox for Q-learning with LunarLander-v3."""

import gymnasium as gym
import numpy as np
import pygame


ACTIONS = ("Do nothing", "Left engine", "Main engine", "Right engine")
BIN_EDGES = ((-0.25, 0.25), (-0.2, 0.2), (-0.15, 0.15))
LEARNING_RATE = 0.5
DISCOUNT_FACTOR = 0.9
SEED = 7
WIDTH, HEIGHT = 900, 570
LANDER_RECT = pygame.Rect(0, 65, 600, 400)


def discretize(observation: np.ndarray) -> tuple[int, int, int]:
    """Bin horizontal position, vertical speed, and tilt into 27 states."""
    selected = (observation[0], observation[3], observation[4])
    return tuple(
        int(np.digitize(value, boundaries))
        for value, boundaries in zip(selected, BIN_EDGES)
    )


def update_q_value(
    old_value: float,
    reward: float,
    best_next_value: float,
    terminal: bool,
) -> float:
    """Apply one Q-learning update using the fixed alpha and gamma."""
    future_value = 0.0 if terminal else DISCOUNT_FACTOR * best_next_value
    target = reward + future_value
    return old_value + LEARNING_RATE * (target - old_value)


def draw_text(screen: pygame.Surface, font: pygame.font.Font, text: str,
              position: tuple[int, int], color=(30, 30, 30)) -> None:
    screen.blit(font.render(text, True, color), position)


def main() -> None:
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("LunarLander: Q-learning sandbox")
    title_font = pygame.font.Font(None, 30)
    font = pygame.font.Font(None, 24)
    small_font = pygame.font.Font(None, 20)
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    q_table: dict[tuple[int, ...], np.ndarray] = {}

    def reset() -> tuple[np.ndarray, tuple[int, int, int]]:
        observation, _ = env.reset(seed=SEED)
        state = discretize(observation)
        return observation, state

    observation, state = reset()
    total_score = 0.0
    step = 0
    last_reward: float | None = None
    last_update = "Take one step. Watch the chosen action’s value change."
    last_action = "None"
    ended = False
    running = True
    clock = pygame.time.Clock()

    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q)
                ):
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        observation, state = reset()
                        total_score = 0.0
                        step = 0
                        last_reward = None
                        last_action = "None"
                        ended = False
                        last_update = "Restarted. The table remembers what it learned this run."
                    elif event.key in (pygame.K_LEFT, pygame.K_RIGHT, pygame.K_UP, pygame.K_SPACE) and not ended:
                        if event.key == pygame.K_LEFT:
                            action = 1
                        elif event.key == pygame.K_RIGHT:
                            action = 3
                        else:
                            action = 2
                        values = q_table.setdefault(
                            state, np.zeros(len(ACTIONS), dtype=np.float64)
                        )
                        old_value = float(values[action])
                        next_observation, reward, terminated, truncated, _ = env.step(action)
                        terminal = terminated or truncated
                        next_state = discretize(next_observation)
                        next_values = q_table.setdefault(
                            next_state, np.zeros(len(ACTIONS), dtype=np.float64)
                        )
                        best_next = 0.0 if terminal else float(np.max(next_values))
                        new_value = update_q_value(old_value, float(reward), best_next, terminal)
                        values[action] = new_value
                        last_update = f"{ACTIONS[action]} value: {old_value:+.2f}  →  {new_value:+.2f}"
                        last_action = ACTIONS[action]
                        last_reward = float(reward)
                        total_score += float(reward)
                        step += 1
                        observation, state = next_observation, next_state
                        ended = terminal
                        if ended:
                            result = "Safe landing!" if terminated and reward > 0 else "Episode ended."
                            last_update += f" | {result} Press R to try again."

            screen.fill((247, 247, 247))
            draw_text(screen, title_font, "Goal: see how experience changes action values", (14, 17))
            draw_text(screen, small_font, "Press a control once, then look at the value that changed.", (14, 43))

            # Gymnasium returns (width, height, RGB); Pygame's surface axes need to be swapped.
            frame = pygame.surfarray.make_surface(np.transpose(env.render(), (1, 0, 2)))
            frame = pygame.transform.scale(frame, LANDER_RECT.size)
            screen.blit(frame, LANDER_RECT.topleft)

            panel_x = 620
            draw_text(screen, title_font, "Action values", (panel_x, 78))
            draw_text(screen, small_font, "For this kind of situation:", (panel_x, 111))
            values = q_table.get(state, np.zeros(len(ACTIONS), dtype=np.float64))
            for index, (action_name, value) in enumerate(zip(ACTIONS, values)):
                color = (35, 115, 65) if action_name == last_action and step else (30, 30, 30)
                draw_text(screen, font, f"{action_name}: {value:+.2f}",
                          (panel_x, 151 + index * 38), color)

            pygame.draw.line(screen, (190, 190, 190), (14, 475), (WIDTH - 14, 475), 1)
            draw_text(screen, title_font, "What changed?", (14, 487))
            reward_text = "--" if last_reward is None else f"{last_reward:+.2f}"
            draw_text(screen, font, f"You chose {last_action}. Reward: {reward_text}.", (14, 523))
            draw_text(screen, font, last_update, (14, 550))
            draw_text(screen, small_font,
                      "LEFT/RIGHT: steer | UP/SPACE: main engine | R: restart | Q/Esc: quit",
                      (14, 590))
            pygame.display.flip()
            clock.tick(30)
    finally:
        env.close()
        pygame.quit()


if __name__ == "__main__":
    main()

"""Interactive visual sandbox for an epsilon-greedy Q-learning agent in LunarLander-v3."""

import random

import gymnasium as gym
import numpy as np
import pygame


ACTIONS = ("Do nothing", "Left engine", "Main engine", "Right engine")
BIN_EDGES = ((-0.25, 0.25), (-0.2, 0.2), (-0.15, 0.15))
LEARNING_RATE = 0.5
DISCOUNT_FACTOR = 0.9
EPSILON = 0.2
AUTO_STEP_MS = 250
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

    rng = random.Random(SEED + 1)

    def reset() -> tuple[np.ndarray, tuple[int, int, int]]:
        observation, _ = env.reset(seed=SEED)
        state = discretize(observation)
        return observation, state

    observation, state = reset()
    total_score = 0.0
    step = 0
    last_reward: float | None = None
    last_update = "Take one step. Watch the chosen action's value change."
    last_action = "None"
    decision_state = state
    ended = False
    auto_mode = True
    next_auto_step = pygame.time.get_ticks()
    last_choice_type = "Agent is choosing"
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
                    if event.key == pygame.K_a:
                        auto_mode = not auto_mode
                        next_auto_step = pygame.time.get_ticks()
                        last_choice_type = "Agent running" if auto_mode else "Agent paused"
                    elif event.key == pygame.K_r:
                        observation, state = reset()
                        total_score = 0.0
                        step = 0
                        last_reward = None
                        last_action = "None"
                        decision_state = state
                        ended = False
                        auto_mode = True
                        next_auto_step = pygame.time.get_ticks()
                        last_update = "Restarted; the table keeps its experience."
                        last_choice_type = "Agent running"
                    elif event.key in (pygame.K_LEFT, pygame.K_RIGHT, pygame.K_UP, pygame.K_SPACE) and not ended:
                        auto_mode = False
                        last_choice_type = "Manual control"
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
                        decision_state = state
                        next_observation, reward, terminated, truncated, _ = env.step(action)
                        terminal = terminated or truncated
                        next_state = discretize(next_observation)
                        next_values = q_table.setdefault(
                            next_state, np.zeros(len(ACTIONS), dtype=np.float64)
                        )
                        best_next = 0.0 if terminal else float(np.max(next_values))
                        new_value = update_q_value(old_value, float(reward), best_next, terminal)
                        values[action] = new_value
                        last_update = f"{ACTIONS[action]} value: {old_value:+.2f}  ->  {new_value:+.2f}"
                        last_action = ACTIONS[action]
                        last_reward = float(reward)
                        total_score += float(reward)
                        step += 1
                        observation, state = next_observation, next_state
                        ended = terminal
                        if ended:
                            auto_mode = False
                            result = "Safe landing" if terminated and reward > 0 else "Episode ended"
                            last_choice_type = result

            now = pygame.time.get_ticks()
            if auto_mode and not ended and now >= next_auto_step:
                values = q_table.setdefault(state, np.zeros(len(ACTIONS), dtype=np.float64))
                if rng.random() < EPSILON:
                    action = rng.randrange(len(ACTIONS))
                    last_choice_type = "Random exploration"
                else:
                    best_actions = np.flatnonzero(values == np.max(values)).tolist()
                    action = rng.choice(best_actions)
                    last_choice_type = "Best Q-value"

                old_value = float(values[action])
                decision_state = state
                next_observation, reward, terminated, truncated, _ = env.step(action)
                terminal = terminated or truncated
                next_state = discretize(next_observation)
                next_values = q_table.setdefault(next_state, np.zeros(len(ACTIONS), dtype=np.float64))
                best_next = 0.0 if terminal else float(np.max(next_values))
                new_value = update_q_value(old_value, float(reward), best_next, terminal)
                values[action] = new_value
                last_update = f"{ACTIONS[action]} value: {old_value:+.2f}  ->  {new_value:+.2f}"
                last_action = ACTIONS[action]
                last_reward = float(reward)
                total_score += float(reward)
                step += 1
                observation, state = next_observation, next_state
                ended = terminal
                next_auto_step = now + AUTO_STEP_MS
                if ended:
                    auto_mode = False
                    result = "Safe landing" if terminated and reward > 0 else "Episode ended"
                    last_choice_type = result

            screen.fill((247, 247, 247))
            draw_text(screen, title_font, "Goal: see how experience changes action values", (14, 17))
            draw_text(screen, small_font, "Agent starts on: it usually uses the table, and sometimes explores randomly.", (14, 43))

            # Gymnasium returns (width, height, RGB); Pygame's surface axes need to be swapped.
            frame = pygame.surfarray.make_surface(np.transpose(env.render(), (1, 0, 2)))
            frame = pygame.transform.scale(frame, LANDER_RECT.size)
            screen.blit(frame, LANDER_RECT.topleft)

            panel_x = 620
            draw_text(screen, title_font, "Values for last decision", (panel_x, 78))
            draw_text(screen, small_font, f"Situation {decision_state}", (panel_x, 111))
            values = q_table.get(decision_state, np.zeros(len(ACTIONS), dtype=np.float64))
            for index, (action_name, value) in enumerate(zip(ACTIONS, values)):
                color = (35, 115, 65) if action_name == last_action and step else (30, 30, 30)
                draw_text(screen, font, f"{action_name}: {value:+.2f}",
                          (panel_x, 151 + index * 38), color)

            pygame.draw.line(screen, (190, 190, 190), (14, 475), (WIDTH - 14, 475), 1)
            draw_text(screen, title_font, "What changed?", (14, 487))
            reward_text = "--" if last_reward is None else f"{last_reward:+.2f}"
            draw_text(screen, font, f"Chosen action: {last_action}. Reward: {reward_text}.", (14, 523))
            draw_text(screen, font, last_update, (14, 550))
            draw_text(screen, font, last_choice_type, (panel_x, 345), (50, 80, 150))
            draw_text(screen, small_font,
                      "A: pause agent | arrows: manual | UP/SPACE: main | R: restart | Q: quit",
                      (14, 590))
            pygame.display.flip()
            clock.tick(30)
    finally:
        env.close()
        pygame.quit()


if __name__ == "__main__":
    main()

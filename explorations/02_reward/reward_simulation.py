"""Compare default LunarLander rewards for idle, random, and manual policies."""

import math

import numpy as np
import pygame
import gymnasium as gym


SEED = 7
MAX_STEPS = 200
STEP_INTERVAL_MS = 100
PANEL_WIDTH = 400
PANEL_HEIGHT = 620
LANDER_SIZE = (400, 267)
PANELS = ("Do nothing", "Random policy", "You control this lander")
ACTION_NAMES = (
    "0 - do nothing",
    "1 - left side engine",
    "2 - main engine",
    "3 - right side engine",
)
TERM_NAMES = {
    "position": "Position (closer is better)",
    "speed": "Speed (slower is better)",
    "upright": "Upright (less tilt is better)",
    "legs": "Leg contact",
}


def reward_terms(observation: np.ndarray) -> dict[str, float]:
    """Return the components used by Gymnasium's shaping reward."""
    x, y, vx, vy, angle, _, left_contact, right_contact = observation
    return {
        "position": -100.0 * math.hypot(x, y),
        "speed": -100.0 * math.hypot(vx, vy),
        "upright": -100.0 * abs(angle),
        "legs": 10.0 * (left_contact + right_contact),
    }


def engine_cost(action: int) -> float:
    if action == 2:
        return -0.30
    if action in (1, 3):
        return -0.03
    return 0.0


def main() -> None:
    pygame.init()
    screen = pygame.display.set_mode((PANEL_WIDTH * len(PANELS), PANEL_HEIGHT))
    pygame.display.set_caption("LunarLander: reward comparison")
    title_font = pygame.font.Font(None, 24)
    font = pygame.font.Font(None, 20)
    score_font = pygame.font.Font(None, 38)
    summary_font = pygame.font.Font(None, 23)
    clock = pygame.time.Clock()
    envs = [gym.make("LunarLander-v3", render_mode="rgb_array") for _ in PANELS]
    states: list[dict] = []

    def reset() -> None:
        states.clear()
        for env in envs:
            observation, _ = env.reset(seed=SEED)
            if len(states) == 1:
                env.action_space.seed(SEED + 1)
            states.append(
                {
                    "observation": observation,
                    # reset() returns a state after an internal no-op step; Gymnasium
                    # already stores the shaping score for that observation.
                    "previous_terms": reward_terms(observation),
                    "delta_terms": {name: 0.0 for name in TERM_NAMES},
                    "engine_cost": 0.0,
                    "reward": None,
                    "score": 0.0,
                    "action": 0,
                    "done": False,
                    "ending": "In progress",
                }
            )

    def draw() -> None:
        screen.fill((245, 245, 245))
        scores = [state["score"] for state in states]
        highest = max(scores)
        leaders = [i for i, score in enumerate(scores) if abs(score - highest) < 0.01]
        both_done = all(state["done"] for state in states)
        limit_reached = step_count >= MAX_STEPS
        prefix = (
            "FINAL SCORES"
            if both_done
            else "SCORES AT STEP LIMIT"
            if limit_reached
            else "LIVE LEADER"
        )
        if len(leaders) > 1:
            summary = f"{prefix}: TIED"
        else:
            leader_index = leaders[0]
            runner_up = max(score for i, score in enumerate(scores) if i != leader_index)
            leader = PANELS[leader_index]
            margin = highest - runner_up
            suffix = " (still running)" if not both_done and not limit_reached else ""
            summary = f"{prefix}: {leader} by {margin:.1f} points{suffix}"
        screen.blit(summary_font.render(summary, True, (25, 25, 25)), (12, 4))

        for index, (label, env, state) in enumerate(zip(PANELS, envs, states)):
            x = index * PANEL_WIDTH
            pixels = np.transpose(env.render(), (1, 0, 2))
            image = pygame.surfarray.make_surface(pixels)
            image = pygame.transform.scale(image, LANDER_SIZE)
            screen.blit(image, (x, 52))

            action_line = f"Step {step_count}/{MAX_STEPS} | action: {ACTION_NAMES[state['action']]}"
            reward_line = (
                "Gym reward this step: --"
                if state["reward"] is None
                else f"Gym reward this step: {state['reward']:+.2f}"
            )
            is_leader = index in leaders
            score_color = (215, 245, 220) if is_leader and len(leaders) == 1 else (232, 232, 232)
            pygame.draw.rect(screen, score_color, (x + 8, 327, PANEL_WIDTH - 16, 44))
            screen.blit(title_font.render(label, True, (25, 25, 25)), (x + 8, 27))
            screen.blit(font.render("SCORE", True, (25, 25, 25)), (x + 16, 339))
            score_text = score_font.render(f"{state['score']:+.1f}", True, (15, 15, 15))
            screen.blit(score_text, (x + 100, 329))
            if len(leaders) > 1:
                status = "TIED"
            elif is_leader:
                status = "LEADING"
            else:
                status = "TRAILING"
            screen.blit(font.render(status, True, (25, 25, 25)), (x + 286, 341))

            lines = [
                (action_line, font),
                (reward_line, font),
                ("Reward parts (terminal result overrides):", font),
            ]
            for name, description in TERM_NAMES.items():
                lines.append((f"  {description}: {state['delta_terms'][name]:+.2f}", font))
            lines.append((f"  Engine cost: {state['engine_cost']:+.2f}", font))
            lines.append((f"Episode: {state['ending']}", font))

            y = 379
            for line, line_font in lines:
                rendered = line_font.render(line, True, (25, 25, 25))
                screen.blit(rendered, (x + 12, y))
                y += 22

        prompt = font.render(
            "Click the window | LEFT: side engine | UP/SPACE: main | RIGHT: side engine | R: restart | Q: quit",
            True,
            (25, 25, 25),
        )
        pygame.draw.line(screen, (185, 185, 185), (PANEL_WIDTH, 27), (PANEL_WIDTH, 600), 1)
        pygame.draw.line(screen, (185, 185, 185), (PANEL_WIDTH * 2, 27), (PANEL_WIDTH * 2, 600), 1)
        screen.blit(prompt, (12, 597))
        pygame.display.flip()

    step_count = 0
    try:
        reset()
        last_step_time = pygame.time.get_ticks() - STEP_INTERVAL_MS
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False
                    elif event.key == pygame.K_r:
                        step_count = 0
                        reset()
                        last_step_time = pygame.time.get_ticks() - STEP_INTERVAL_MS

            now = pygame.time.get_ticks()
            if step_count < MAX_STEPS and now - last_step_time >= STEP_INTERVAL_MS:
                step_count += 1
                last_step_time = now
                keys = pygame.key.get_pressed()
                for index, (env, state) in enumerate(zip(envs, states)):
                    action = 0
                    if index == 2:
                        if keys[pygame.K_UP] or keys[pygame.K_SPACE]:
                            action = 2
                        elif keys[pygame.K_LEFT]:
                            action = 1
                        elif keys[pygame.K_RIGHT]:
                            action = 3
                    elif index == 1:
                        action = int(env.action_space.sample())

                    if state["done"]:
                        continue

                    previous_terms = state["previous_terms"]
                    observation, reward, terminated, truncated, _ = env.step(action)
                    current_terms = reward_terms(observation)
                    delta_terms = {
                        name: (
                            current_terms[name] - previous_terms[name]
                            if previous_terms is not None
                            else 0.0
                        )
                        for name in TERM_NAMES
                    }
                    cost = engine_cost(action)

                    state.update(
                        {
                            "observation": observation,
                            "previous_terms": current_terms,
                            "delta_terms": delta_terms,
                            "engine_cost": cost,
                            "reward": float(reward),
                            "score": state["score"] + float(reward),
                            "action": action,
                            "done": terminated or truncated,
                        }
                    )
                    if terminated:
                        state["ending"] = (
                            f"safe landing (+100)" if reward > 0 else "crash / out of bounds (-100)"
                        )
                    elif truncated:
                        state["ending"] = "time limit reached"

                    print(
                        f"step={step_count:3} panel={index + 1} action={action} "
                        f"position={delta_terms['position']:+.2f} "
                        f"speed={delta_terms['speed']:+.2f} "
                        f"upright={delta_terms['upright']:+.2f} "
                        f"legs={delta_terms['legs']:+.2f} cost={cost:+.2f} "
                        f"gym_reward={reward:+.2f} score={state['score']:+.2f}",
                        flush=True,
                    )

            draw()
            clock.tick(60)
    finally:
        for env in envs:
            env.close()
        pygame.quit()


if __name__ == "__main__":
    main()

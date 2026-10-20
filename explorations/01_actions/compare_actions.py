"""Compare two fixed LunarLander actions side by side while they run."""

import numpy as np
import pygame
import gymnasium as gym


SEED = 7
MAX_STEPS = 200
STEP_INTERVAL_MS = 80
PANELS = (("Reference: do nothing", 0), ("You: keyboard control", 0))
FRAME_SIZE = (600, 400)
PANEL_HEIGHT = 530


def main() -> None:
    pygame.init()
    screen = pygame.display.set_mode((FRAME_SIZE[0] * 2, PANEL_HEIGHT))
    pygame.display.set_caption("LunarLander: fixed-action comparison")
    font = pygame.font.Font(None, 26)
    small_font = pygame.font.Font(None, 21)
    clock = pygame.time.Clock()
    envs = [gym.make("LunarLander-v3", render_mode="rgb_array") for _ in PANELS]
    states = []

    def reset() -> None:
        states.clear()
        for env in envs:
            observation, _ = env.reset(seed=SEED)
            states.append(
                {"observation": observation, "reward": 0.0, "done": False, "action": 0}
            )

    def draw() -> None:
        screen.fill((245, 245, 245))
        for index, ((label, _), env, state) in enumerate(zip(PANELS, envs, states)):
            x = index * FRAME_SIZE[0]
            image = pygame.surfarray.make_surface(np.transpose(env.render(), (1, 0, 2)))
            screen.blit(image, (x, 55))
            obs = state["observation"]
            title = font.render(label, True, (20, 20, 20))
            details = small_font.render(
                f"Step {step_count}/{MAX_STEPS} | score {state['reward']:.1f} | "
                f"vy {obs[3]:+.2f} | action {state['action']}",
                True,
                (20, 20, 20),
            )
            screen.blit(title, (x + 12, 12))
            screen.blit(details, (x + 12, 462))
        prompt = small_font.render(
            "Click window | LEFT: left engine, UP/SPACE: main, RIGHT: right engine | R reset | Q quit",
            True,
            (35, 35, 35),
        )
        screen.blit(prompt, (12, 502))
        pygame.display.flip()

    step_count = 0
    try:
        reset()
        last_step_time = pygame.time.get_ticks() - STEP_INTERVAL_MS
        draw()
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
                for index, ((_, _), env, state) in enumerate(zip(PANELS, envs, states)):
                    action = 0
                    if index == 1:
                        if keys[pygame.K_UP] or keys[pygame.K_SPACE]:
                            action = 2
                        elif keys[pygame.K_LEFT]:
                            action = 1
                        elif keys[pygame.K_RIGHT]:
                            action = 3
                    if not state["done"]:
                        obs, reward, terminated, truncated, _ = env.step(action)
                        state["observation"] = obs
                        state["reward"] += reward
                        state["done"] = terminated or truncated
                        state["action"] = action
                for (label, _), state in zip(PANELS, states):
                    print(
                        f"step={step_count:3} {label:24} "
                        f"score={state['reward']:7.2f} "
                        f"vx={state['observation'][2]:+.3f} "
                        f"vy={state['observation'][3]:+.3f} "
                        f"angle={state['observation'][4]:+.3f}",
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

"""Open Gymnasium LunarLander-v3 and drive it with random actions."""

import gymnasium as gym


def main() -> None:
    env = gym.make("LunarLander-v3", render_mode="human")
    try:
        observation, info = env.reset()
        episode_reward = 0.0

        while True:
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

            if terminated or truncated:
                print(f"Episode finished with score: {episode_reward:.1f}")
                observation, info = env.reset()
                episode_reward = 0.0
    except KeyboardInterrupt:
        print("\nDemo stopped.")
    finally:
        env.close()


if __name__ == "__main__":
    main()

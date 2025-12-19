# python/trainer/test_client.py

from trainer.gym_env import SnakeRemoteEnv  # or from .gym_env import SnakeRemoteEnv
import numpy as np

def main():
    env = SnakeRemoteEnv()
    obs, info = env.reset()
    print("Initial obs shape:", obs.shape)
    print("Initial info:", info)

    done = False
    total_reward = 0.0
    steps = 0

    while not done and steps < 20:
        action = np.random.randint(0, 4)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1
        print(f"step={steps}, reward={reward}, done={done}, score={info['score']}")

    print("Episode finished. Total reward:", total_reward)
    env.close()

if __name__ == "__main__":
    main()

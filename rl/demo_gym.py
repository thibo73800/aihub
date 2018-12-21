import gym
import time

env = gym.make('Breakout-v0')
env.reset()

for _ in range(1000):
    env.render()
    observation, reward, done, info = env.step(env.action_space.sample())
    if reward != 0:
        print("Reward", reward)
    time.sleep(0.1)
    if done:
        break

#!/usr/bin/env python3

import numpy as np
import gym
import gym_minigrid
import time

def state_to_key(obs):
    return str(obs["image"].tolist()+[obs["direction"]]).strip()

def update_Q(Q, s, sp, a, r, done):
    if s not in Q:
        Q[s] = np.array([0., 0., 0., 0.])
    if sp not in Q:
        Q[sp] = np.array([0., 0., 0., 0.])

    ap = np.argmax(Q[sp])
    if not done:
        Q[s][a] = Q[s][a] + 0.01*(r + 0.99*Q[sp][ap] - Q[s][a])
    else:
        Q[s][a] = Q[s][a] + 0.01*(r - Q[s][a])

def create_state_if_not_exist(Q, s):
    if s not in Q:
        Q[s] = np.array([0., 0., 0., 0.])

def main():

    Q = {}

    env = gym.make("MiniGrid-Empty-6x6-v0")
    eps = 0.01

    for epoch in range(100):

        s = env.reset()
        s = state_to_key(s)
        done = False

        while not done:

            if np.random.rand() < eps:
                a = np.random.randint(0, 4)
            else:
                create_state_if_not_exist(Q, s)
                a = np.argmax(Q[s])

            sp, r, done, info = env.step(a)
            sp = state_to_key(sp)

            update_Q(Q, s, sp, a, r, done)

            s = sp

        print("eps", eps)
        eps = max(0.1, eps*0.99)


    for epoch in range(100):
        s = env.reset()
        s = state_to_key(s)
        done = False

        while not done:
            create_state_if_not_exist(Q, s)
            a = np.argmax(Q[s])
            sp, r, done, info = env.step(a)
            sp = state_to_key(sp)
            s = sp
            env.render()
            time.sleep(0.1)
        print("r", r)

if __name__ == "__main__":
    main()

import numpy as np
from random import randint
import random

class EnvGrid(object):
    """
        docstring forEnvGrid.
    """
    def __init__(self):
        super(EnvGrid, self).__init__()

        self.grid = [
            [0, 0, 1],
            [0, -1, 0],
            [0, 0, 0]
        ]
        # Starting position
        self.y = 2
        self.x = 0

        self.actions = [
            [-1, 0], # Up
            [1, 0], #Down
            [0, -1], # Left
            [0, 1] # Right
        ]

    def reset(self):
        """
            Reset world
        """
        self.y = 2
        self.x = 0
        return (self.y*3+self.x)

    def step(self, action):
        """
            Action: 0, 1, 2, 3
        """
        self.y = max(0, min(self.y + self.actions[action][0],2))
        self.x = max(0, min(self.x + self.actions[action][1],2))

        return (self.y*3+self.x) , self.grid[self.y][self.x]

    def show(self):
        """
            Show the grid
        """
        print("---------------------")
        y = 0
        for line in self.grid:
            x = 0
            for pt in line:
                print("%s\t" % (pt if y != self.y or x != self.x else "X"), end="")
                x += 1
            y += 1
            print("")

    def is_finished(self):
        return self.grid[self.y][self.x] == 1

def take_action(st, Q, eps):
    # Take an action
    if random.uniform(0, 1) < eps:
        action = randint(0, 3)
    else: # Or greedy action
        action = np.argmax(Q[st])
    return action

if __name__ == '__main__':
    env = EnvGrid()
    st = env.reset()

    Q = [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ]

    for _ in range(100):
        # Reset the game
        st = env.reset()
        while not env.is_finished():
            #env.show()
            #at = int(input("$>"))
            at = take_action(st, Q, 0.4)

            stp1, r = env.step(at)
            #print("s", stp1)
            #print("r", r)

            # Update Q function
            atp1 = take_action(stp1, Q, 0.0)
            Q[st][at] = Q[st][at] + 0.1*(r + 0.9*Q[stp1][atp1] - Q[st][at])

            st = stp1

    for s in range(0, 9):
    for s in range(9):
        print(s, Q[s])

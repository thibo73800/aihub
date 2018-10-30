from random import randint
import random
import numpy as np

class TicTacToeGame(object):
    """
        StickGame.
    """

    def __init__(self):
        # @nb Number of stick to play with
        super(TicTacToeGame, self).__init__()
        self.state = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.finished = False

    def is_finished(self):
        # Check if the game is over @return Boolean
        return self.finished

    def reset(self):
        # Reset the state of the game
        self.state = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.finished = False
        return self.state_to_nb(self.state)

    def display(self):
        # Display the state of the game
        print("State id:%s"%self.state_to_nb(self.state))
        display = "%s %s %s \n%s %s %s \n%s %s %s" % tuple(self.state)
        print(display.replace("1", "O").replace("2", "X").replace("0", "-"))

    def next_actions(self):
        # Return the next possible actions
        actions = []
        for p in range(0, 9):
            if self.state[p] == 0:
                actions.append(p)
        return actions

    def next_possible_state(self, action, p):
        # Return the next possible actions
        state = [v for v in self.state]
        state[action] = p
        return state

    def state_to_nb(self, state):
        return str(state)
        i = 0
        nb = 0
        for p in state:
            nb += p*3**i
            i += 1
        return nb

    def step(self, action, p):
        self.state[action] = p
        st = self.state
        n_actions = self.next_actions()
        if len(n_actions) == 0:
            self.finished = True
        if (st[0] == p and st[1] == p and st[2] == p) or \
        (st[3] == p and st[4] == p and st[5] == p) or \
        (st[6] == p and st[7] == p and st[8] == p) or \
        (st[0] == p and st[3] == p and st[6] == p) or \
        (st[1] == p and st[4] == p and st[7] == p) or \
        (st[2] == p and st[5] == p and st[8] == p) or \
        (st[0] == p and st[4] == p and st[8] == p) or \
        (st[2] == p and st[4] == p and st[6] == p):
            self.finished = True
            return self.state_to_nb(self.state), 1
        else:
            return self.state_to_nb(self.state), 0

class StickPlayer(object):
    """
        Stick Player
    """

    def __init__(self, is_human, p, trainable=True):
        # @nb Number of stick to play with
        super(StickPlayer, self).__init__()
        self.is_human = is_human
        self.history = []
        self.V = {}
        self.p = p
        self.win_nb = 0.
        self.lose_nb = 0.
        self.rewards = []
        self.eps = 0.99
        self.trainable = trainable

    def reset_stat(self):
        # Reset stat
        self.win_nb = 0
        self.lose_nb = 0
        self.rewards = []

    def greedy_step(self, state, game, display_value=False):
        # Greedy step
        actions = game.next_actions()
        vboard = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        vmax = None
        vi = None
        for i in range(0, len(actions)):
            a = actions[i]
            n_state = game.next_possible_state(a, self.p)
            nb = game.state_to_nb(n_state)

            if nb not in self.V:
                self.V[nb] = 0

            vboard[a] = self.V[nb]
            if vmax is None or vmax < self.V[nb]:
                vmax = self.V[nb]
                vi = i

        if display_value:
            display = "%.2f %.2f %.2f \n%.2f %.2f %.2f \n%.2f %.2f %.2f" % tuple(vboard)
            print(display)

        return actions[vi]

    def play(self, state, game, display_value=False):
        # PLay given the @state (int)
        if self.is_human is False:
            # Take random action
            if random.uniform(0, 1) < self.eps:
                action = random.choice(game.next_actions())
            else: # Or greedy action
                action = self.greedy_step(state, game, display_value)
        else:
            action = int(input("$>"))
        return action

    def add_transition(self, n_tuple):
        # Add one transition to the history: tuple (s, a , r, s')
        self.history.append(n_tuple)

    def train(self, display_value):
        if not self.trainable or self.is_human is True:
            return

        # Update the value function if this player is not human
        f = 0
        for transition in reversed(self.history):
            s, r, sp = transition

            if s not in self.V:
                self.V[s] = 0
            if sp not in self.V:
                self.V[sp] = 0

            if f == 0:
                # For the last element, we train it toward the reward
                self.V[sp] = self.V[sp] + 0.01*(r - self.V[sp])
            self.V[s] = self.V[s] + 0.01*(self.V[sp] - self.V[s])
            f += 1


        self.history = []

def play(game, p1, p2, train=True, display_value=False):
    state = game.reset()
    players = [p1, p2]
    random.shuffle(players)
    p = 0
    while game.is_finished() is False:

        if players[0].is_human or players[1].is_human:
            game.display()

        if state not in players[p%2].V:
            players[p%2].V[state] = 0
        action = players[p%2].play(state, game, display_value)
        n_state, reward = game.step(action, players[p%2].p)

        #  Game is over. Ass stat
        if (reward == 1):
            players[p%2].win_nb += 1

        if display_value:
            print("reward", reward)

        players[p%2].add_transition((state, reward, n_state))
        players[(p+1)%2].add_transition((state, reward * -1, n_state))

        state = n_state
        p += 1

    if train:
        p1.train(display_value=display_value)
        p2.train(display_value=display_value)

if __name__ == '__main__':
    game = TicTacToeGame()

    # Players to train
    p1 = StickPlayer(is_human=False, p=1, trainable=True)
    p2 = StickPlayer(is_human=False, p=2, trainable=True)
    # Human player and random player
    human = StickPlayer(is_human=True, p=2, trainable=False)
    random_player = StickPlayer(is_human=False, p=2, trainable=False)

    # Train the agent
    for i in range(0, 100000):
        if i % 10 == 0:
            p1.eps = max(0.05, p1.eps*0.999)
            p2.eps = max(0.05, p2.eps*0.999)
        if i % 1000 == 0:
            p1.reset_stat()
            # Play agains a random player
            for _ in range(0, 100):
                play(game, p1, random_player, train=False)
            print("eps=%sp1 win rate=%s" % (p1.eps, p1.win_nb/100.))

        play(game, p1, p2)

    p1.eps = 0.0
    p1.reset_stat()
    # Play agains a random player
    for _ in range(0, 10000):
        play(game, p1, random_player, train=False)
    print("eps=%sp1 win rate=%s" % (p1.eps, p1.win_nb/10000.))

    # Play agains us
    while True:
        play(game, p1, human, train=True, display_value=True)

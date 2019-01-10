from random import randint
import tensorflow as tf
import numpy as np
import scipy.signal

"""
    Exemple of the Policy Gradient Algorithm
"""

class Buffer:

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(Buffer.combined_shape(size, obs_dim), dtype=np.float32)
        # Actions buffer
        self.act_buf = np.zeros(size, dtype=np.float32)
        # Advantages buffer
        self.adv_buf = np.zeros(size, dtype=np.float32)
        # Value function buffer
        self.val_buf = np.zeros(size, dtype=np.float32)
        # Rewards buffer
        self.rew_buf = np.zeros(size, dtype=np.float32)
        # Log probability of action a with the policy
        self.logp_buf = np.zeros(size, dtype=np.float32)
        # Gamma and lam to compute the advantage
        self.gamma, self.lam = gamma, lam
        # Rreturn buffer (used to train the value fucntion)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        # ptr: Position to insert the next tuple
        # path_start_idx Posittion of the current trajectory
        # max_size Max size of the buffer
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    @staticmethod
    def discount_cumsum(x, discount):
        """
            x = [x0, x1, x2]
            output: [x0 + discount * x1 + discount^2 * x2, x1 + discount * x2, x2]
        """
        return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

    @staticmethod
    def combined_shape(length, shape=None):
        if shape is None:
            return (length,)
        return (length, shape) if np.isscalar(shape) else (length, *shape)

    def store(self, obs, act, rew, logp, val):
        # Append one timestep of agent-environment interaction to the buffer.
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.logp_buf[self.ptr] = logp
        self.val_buf[self.ptr] = val
        self.ptr += 1

    def finish_path(self, last_val=0):
        # Select the path
        path_slice = slice(self.path_start_idx, self.ptr)
        # Append the last_val to the trajectory
        rews = np.append(self.rew_buf[path_slice], last_val)
        # Append the last value to the value function
        vals = np.append(self.val_buf[path_slice], last_val)
        # Deltas = r + y*v(s') - v(s)
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = Buffer.discount_cumsum(deltas, self.gamma)
        # Advantage
        self.ret_buf[path_slice] = Buffer.discount_cumsum(rews, self.gamma)[:-1]
        self.path_start_idx = self.ptr

    def get(self):
        assert self.ptr == self.max_size # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        # Normalize the Advantage
        if np.std(self.adv_buf) != 0:
            self.adv_buf = (self.adv_buf - np.mean(self.adv_buf)) / np.std(self.adv_buf)
        return self.obs_buf, self.act_buf, self.adv_buf, self.logp_buf, self.ret_buf


class ActorCritic(object):
    """
        Implementation of Policy gradient algorithm
    """
    def __init__(self, input_space, action_space, pi_lr, v_lr, buffer_size, seed):
        super(ActorCritic, self).__init__()

        # Stored the spaces
        self.input_space = input_space
        self.action_space = action_space
        self.seed = seed
        # NET Buffer defined above
        self.buffer = Buffer(
            obs_dim=input_space,
            act_dim=action_space,
            size=buffer_size
        )
        # Learning rate of the policy network and the value network
        self.pi_lr = pi_lr
        self.v_lr = v_lr
        # The tensorflow session (set later)
        self.sess = None
        # Apply a random seed on tensorflow and numpy
        tf.set_random_seed(42)
        np.random.seed(42)

    def compile(self):
        """
            Compile the model
        """
        # tf_a : Input: Chosen action
        # tf_map: Input: Input state
        # tf_tv: Input: Target value function
        # tf_adv: Input: Advantage
        self.tf_map, self.tf_a, self.tf_adv, self.tf_tv = ActorCritic.inputs(
            map_space=self.input_space,
            action_space=self.action_space
        )
        # mu_op: Used to get the exploited prediction of the model
        # pi_op: Used to get the prediction of the model
        # logp_a_op: Used to get the log likelihood of taking action a with the current policy
        # logp_pi_op: Used to get the log likelihood of the predicted action @pi_op
        # v_op: Used to get the value function of the given state
        self.mu_op, self.pi_op, self.logp_a_op, self.logp_pi_op, self.v_op = ActorCritic.mlp(
            tf_map=self.tf_map,
            tf_a=self.tf_a,
            action_space=self.action_space,
            seed=self.seed
        )
        # Error
        self.pi_loss, self.v_loss = ActorCritic.net_objectives(
            tf_adv=self.tf_adv,
            logp_a_op=self.logp_a_op,
            v_op=self.v_op,
            tf_tv=self.tf_tv
        )
        # Optimization
        self.train_pi = tf.train.AdamOptimizer(learning_rate=self.pi_lr).minimize(self.pi_loss)
        self.train_v = tf.train.AdamOptimizer(learning_rate=self.v_lr).minimize(self.v_loss)
        # Entropy
        self.approx_ent = tf.reduce_mean(-self.logp_a_op)


    def set_sess(self, sess):
        # Set the tensorflow used to run this model
        self.sess = sess

    def step(self, states):
        # Take actions given the states
        # Return mu (policy without exploration), pi (policy with the current exploration) and
        # the log probability of the action chossen by pi
        mu, pi, logp_pi, v = self.sess.run([self.mu_op, self.pi_op, self.logp_pi_op, self.v_op], feed_dict={
            self.tf_map: states
        })
        return mu, pi, logp_pi, v

    def store(self, obs, act, rew, logp, val):
        # Store the observation, action, reward, log probability of the action and state value
        # into the buffer
        self.buffer.store(obs, act, rew, logp, val)

    def finish_path(self, last_val=0):
        self.buffer.finish_path(last_val=last_val)

    def train(self, additional_infos={}):
        # Get buffer
        obs_buf, act_buf, adv_buf, logp_last_buf, ret_buf = self.buffer.get()
        # Train the model
        pi_loss_list = []
        entropy_list = []
        v_loss_list = []

        for step in range(5):
            _, entropy, pi_loss = self.sess.run([self.train_pi, self.approx_ent, self.pi_loss], feed_dict= {
                self.tf_map: obs_buf,
                self.tf_a:act_buf,
                self.tf_adv: adv_buf
            })
            entropy_list.append(entropy)
            pi_loss_list.append(pi_loss)


        for step in range(5):
            _, v_loss = self.sess.run([self.train_v, self.v_loss], feed_dict= {
                self.tf_map: obs_buf,
                self.tf_tv: ret_buf,
            })

            v_loss_list.append(v_loss)

        print("Entropy : %s" % (np.mean(entropy_list)), end="\r")


    @staticmethod
    def gaussian_likelihood(x, mu, log_std):
        # Compute the gaussian likelihood of x with a normal gaussian distribution of mean @mu
        # and a std @log_std
        pre_sum = -0.5 * (((x-mu)/(tf.exp(log_std)+1e-8))**2 + 2*log_std + np.log(2*np.pi))
        return tf.reduce_sum(pre_sum, axis=1)

    @staticmethod
    def inputs(map_space, action_space):
        """
            @map_space Tuple of the space. Ex (size,)
            @action_space Tuple describing the action space. Ex (size,)
        """
        # Map of the game
        tf_map = tf.placeholder(tf.float32, shape=(None, *map_space), name="tf_map")
        # Possible actions (Should be two: x,y for the beacon game)
        tf_a = tf.placeholder(tf.int32, shape=(None,), name="tf_a")
        # Advantage
        tf_adv = tf.placeholder(tf.float32, shape=(None,), name="tf_adv")
        # Target value
        tf_tv = tf.placeholder(tf.float32, shape=(None,), name="tf_tv")
        return tf_map, tf_a, tf_adv, tf_tv

    @staticmethod
    def mlp(tf_map, tf_a, action_space, seed=None):
        if seed is not None:
            tf.random.set_random_seed(seed)

        # Expand the dimension of the input
        tf_map_expand = tf.expand_dims(tf_map, axis=3)

        flatten = tf.layers.flatten(tf_map_expand)
        hidden = tf.layers.dense(flatten, units=256, activation=tf.nn.relu)

        # Logits policy
        spacial_action_logits = tf.layers.dense(hidden, units=action_space, activation=None)
        # Logits value function
        v_op = tf.layers.dense(hidden, units=1, activation=None)

        # Add take the log of the softmax
        logp_all = tf.nn.log_softmax(spacial_action_logits)
        # Take random actions according to the logits (Exploration)
        pi_op = tf.squeeze(tf.multinomial(spacial_action_logits,1), axis=1)
        mu = tf.argmax(spacial_action_logits, axis=1)

        # Gives log probability, according to  the policy, of taking actions @a in states @x
        logp_a_op = tf.reduce_sum(tf.one_hot(tf_a, depth=action_space) * logp_all, axis=1)
        # Gives log probability, according to the policy, of the action sampled by pi.
        logp_pi_op = tf.reduce_sum(tf.one_hot(pi_op, depth=action_space) * logp_all, axis=1)

        return mu, pi_op, logp_a_op, logp_pi_op, v_op

    @staticmethod
    def net_objectives(logp_a_op, tf_adv, v_op, tf_tv, clip_ratio=0.2):
        """
            @v_op: Predicted value function
            @tf_tv: Expected value function
            @logp_a_op: Log likelihood of taking action under the current policy
            @tf_logp_old_pi: Log likelihood of the last policy
            @tf_adv: Advantage input
        """
        pi_loss = -tf.reduce_mean(logp_a_op*tf_adv)
        v_loss = tf.reduce_mean((tf_tv - v_op)**2)
        return pi_loss, v_loss

class GridWorld(object):
    """
        docstring for GridWorld.
    """
    def __init__(self):
        super(GridWorld, self).__init__()

        self.rewards = [
            [0,  0,  0, 0, -1, 0, 0],
            [0, -1, -1, 0, -1, 0, 0],
            [0, -1, -1, 1, -1, 0, 0],
            [0, -1, -1, 0, -1, 0, 0],
            [0,  0,  0, 0,  0, 0, 0],
            [0,  0,  0, 0,  0, 0, 0],
            [0,  0,  0, 0,  0, 0, 0],
        ]
        self.position = [6, 6] # y, x

    def gen_state(self):
        # Generate a state given the current position of the agent
        state = np.zeros((7, 7))
        state[self.position[0]][self.position[1]] = 1
        return state

    def step(self, action):
        if action == 0: # Top
            self.position = [(self.position[0] - 1) % 7, self.position[1]]
        elif action == 1: # Left
            self.position = [self.position[0], (self.position[1] - 1) % 7]
        elif action == 2: # Right
            self.position = [self.position[0], (self.position[1] + 1) % 7]
        elif action == 3: # Down
            self.position = [(self.position[0] + 1) % 7, self.position[1]]

        reward = self.rewards[self.position[0]][self.position[1]]
        done = False if reward == 0 else True
        state = self.gen_state()
        if done: # The agent is dead, reset the game
            self.position = [6, 6]
        return state, reward, done

    def display(self):
        y = 0
        print("="*14)
        for line in self.rewards:
            x = 0
            for case in line:
                if case == -1:
                    c = "0"
                elif (y == self.position[0] and x == self.position[1]):
                    c = "A"
                elif case == 1:
                    c = "T"
                else:
                    c = "-"
                print(c, end=" ")
                x += 1
            y += 1
            print()

def main():
    grid = GridWorld()
    buffer_size = 1000

    # Create the NET class
    agent = ActorCritic(
    	input_space=(7, 7),
    	action_space=4,
    	pi_lr=0.001,
        v_lr=0.001,
    	buffer_size=buffer_size,
    	seed=42
    )
    agent.compile()
    # Init Session
    sess = tf.Session()
    # Init variables
    sess.run(tf.global_variables_initializer())
    # Set the session
    agent.set_sess(sess)

    rewards = []

    b = 0

    for epoch in range(10000):

        done = False
        state = grid.gen_state()

        while not done:
            _, pi, logpi, v = agent.step([state])
            n_state, reward, done = grid.step(pi[0])
            agent.store(state, pi[0], reward, logpi, v)
            b += 1

            state = n_state

            if done:
                agent.finish_path(reward)
                rewards.append(reward)
                if len(rewards) > 1000:
                    rewards.pop(0)
            if b == buffer_size:
                if not done:
                    # Bootstrap the last value
                    agent.finish_path(v)
                agent.train()
                b = 0

        if epoch % 1000 == 0:
            print("Rewards mean:%s" % np.mean(rewards))

    for epoch in range(10):
        import time
        print("=========================TEST=================================")
        done = False
        state = grid.gen_state()

        while not done:
            time.sleep(1)
            _, pi, logpi, v = agent.step([state])
            n_state, _, done = grid.step(pi[0])
            print("v", v)
            grid.display()
            state = n_state
        print("reward=>", reward)

if __name__ == '__main__':
    main()

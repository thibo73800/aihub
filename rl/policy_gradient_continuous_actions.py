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
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
        # Advantages buffer
        self.adv_buf = np.zeros(size, dtype=np.float32)
        # Rewards buffer
        self.rew_buf = np.zeros(size, dtype=np.float32)
        # Log probability of action a with the policy
        self.logp_buf = np.zeros(size, dtype=np.float32)
        # Gamma and lam to compute the advantage
        self.gamma, self.lam = gamma, lam
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

    def store(self, obs, act, rew, logp):
        """
            Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        # Select the path
        path_slice = slice(self.path_start_idx, self.ptr)
        # Append the last_val to the trajectory
        rews = np.append(self.rew_buf[path_slice], last_val)
        # Advantage
        self.adv_buf[path_slice] = Buffer.discount_cumsum(rews, self.gamma)[:-1]
        self.path_start_idx = self.ptr

    def get(self):
        assert self.ptr == self.max_size # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        # Normalize the Advantage
        if np.std(self.adv_buf) != 0:
            self.adv_buf = (self.adv_buf - np.mean(self.adv_buf)) / np.std(self.adv_buf)
        return self.obs_buf, self.act_buf, self.adv_buf, self.logp_buf


class PolicyGradient(object):
    """
        Implementation of Policy gradient algorithm
    """
    def __init__(self, input_space, action_space, pi_lr, buffer_size, seed):
        super(PolicyGradient, self).__init__()

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
        # Learning rate of the policy network
        self.pi_lr = pi_lr
        # The tensorflow session (set later)
        self.sess = None
        # Apply a random seed on tensorflow and numpy
        tf.set_random_seed(42)
        np.random.seed(42)

    def compile(self):
        """
            Compile the model
        """
        # tf_map: Input: Input state
        # tf_adv: Input: Advantage
        self.tf_map, self.tf_a, self.tf_adv = PolicyGradient.inputs(
            map_space=self.input_space,
            action_space=self.action_space
        )
        # mu_op: Used to get the exploited prediction of the model
        # pi_op: Used to get the prediction of the model
        # logp_a_op: Used to get the log likelihood of taking action a with the current policy
        # logp_pi_op: Used to get the log likelihood of the predicted action @pi_op
        # log_std: Used to get the currently used log_std
        self.mu_op, self.pi_op, self.logp_a_op, self.logp_pi_op, self.std = PolicyGradient.mlp(
            tf_map=self.tf_map,
            tf_a=self.tf_a,
            action_space=self.action_space,
            seed=self.seed
        )
        # Error
        self.pi_loss = PolicyGradient.net_objectives(
            tf_adv=self.tf_adv,
            logp_a_op=self.logp_a_op
        )
        # Optimization
        self.train_pi = tf.train.AdamOptimizer(learning_rate=self.pi_lr).minimize(self.pi_loss)
        # Entropy
        self.approx_ent = tf.reduce_mean(-self.logp_a_op)


    def set_sess(self, sess):
        # Set the tensorflow used to run this model
        self.sess = sess

    def step(self, states):
        # Take actions given the states
        # Return mu (policy without exploration), pi (policy with the current exploration) and
        # the log probability of the action chossen by pi
        mu, pi, logp_pi = self.sess.run([self.mu_op, self.pi_op, self.logp_pi_op], feed_dict={
            self.tf_map: states
        })
        return mu, pi, logp_pi

    def store(self, obs, act, rew, logp):
        # Store the observation, action, reward and the log probability of the action
        # into the buffer
        self.buffer.store(obs, act, rew, logp)

    def finish_path(self, last_val=0):
        self.buffer.finish_path(last_val=last_val)

    def train(self, additional_infos={}):
        # Get buffer
        obs_buf, act_buf, adv_buf, logp_last_buf = self.buffer.get()
        # Train the model
        pi_loss_list = []
        entropy_list = []

        import time
        t = time.time()

        for step in range(5):
            _, entropy, pi_loss, std = self.sess.run([self.train_pi, self.approx_ent, self.pi_loss, self.std], feed_dict= {
                self.tf_map: obs_buf,
                self.tf_a:act_buf,
                self.tf_adv: adv_buf
            })

            pi_loss_list.append(pi_loss)
            entropy_list.append(entropy)

        print("Std: %.2f, Entropy : %.2f" % (std[0], np.mean(entropy_list)), end="\r")


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
        tf_a = tf.placeholder(tf.float32, shape=(None, action_space), name="tf_a")
        # Advantage
        tf_adv = tf.placeholder(tf.float32, shape=(None,), name="tf_adv")
        return tf_map, tf_a, tf_adv

    @staticmethod
    def gaussian_likelihood(x, mu, log_std):
        # Compute the gaussian likelihood of x with a normal gaussian distribution of mean @mu
        # and a std @log_std
        pre_sum = -0.5 * (((x-mu)/(tf.exp(log_std)+1e-8))**2 + 2*log_std + np.log(2*np.pi))
        return tf.reduce_sum(pre_sum, axis=1)
        #pre_sum = (1.*tf.exp((-(x-mu)**2)/(2*std**2)))/tf.sqrt(2*3.14*std**2)
        #retun

    @staticmethod
    def mlp(tf_map, tf_a, action_space, seed=None):
        if seed is not None:
            tf.random.set_random_seed(seed)

        # Expand the dimension of the input
        tf_map_expand = tf.expand_dims(tf_map, axis=3)

        flatten = tf.layers.flatten(tf_map_expand)
        hidden = tf.layers.dense(flatten, units=256, activation=tf.nn.relu)
        mu = tf.layers.dense(hidden, units=action_space, activation=tf.nn.relu)

        log_std = tf.get_variable(name='log_std', initializer=-0.5*np.ones(action_space, dtype=np.float32))
        std = tf.exp(log_std)
        pi_op = mu + tf.random_normal(tf.shape(mu)) * std

        # Gives log probability, according to  the policy, of taking actions @a in states @x
        logp_a_op = PolicyGradient.gaussian_likelihood(tf_a, mu, log_std)
        # Gives log probability, according to the policy, of the action sampled by pi.
        logp_pi_op = PolicyGradient.gaussian_likelihood(pi_op, mu, log_std)

        return mu, pi_op, logp_a_op, logp_pi_op, std

    @staticmethod
    def net_objectives(logp_a_op, tf_adv, clip_ratio=0.2):
        """
            @v_op: Predicted value function
            @tf_tv: Expected advantage
            @logp_a_op: Log likelihood of taking action under the current policy
            @tf_logp_old_pi: Log likelihood of the last policy
            @tf_adv: Advantage input
        """
        pi_loss = -tf.reduce_mean(logp_a_op*tf_adv)
        return pi_loss

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

    def step(self, y, x):
        # y and x coordinates

        # Move in the direction of the "click"
        self.position = [
            self.position[0] + min(1, max(-1, (y - self.position[0]))),
            self.position[1] + min(1, max(-1, (x - self.position[1])))
        ]
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
    agent = PolicyGradient(
    	input_space=(7, 7),
    	action_space=2,
    	pi_lr=0.001,
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
    display = False

    for epoch in range(100000):

        done = False
        state = grid.gen_state()

        s = 0
        while not done and s < 20:
            s += 1
            _, pi, logpi = agent.step([state])

            y = max(0, min(6, int(round((pi[0][0]+1.) / 2*6))))
            x = max(0, min(6, int(round((pi[0][1]+1.) / 2*6))))

            if display:
                import time
                time.sleep(0.1)
                grid.display()

            n_state, reward, done = grid.step(y, x)
            agent.store(state, pi[0], -0.1 if reward == 0 else reward, logpi)
            b += 1

            state = n_state

            if done:
                agent.finish_path(reward)
                rewards.append(reward)
                if len(rewards) > 1000:
                    rewards.pop(0)
            if b == buffer_size:
                if not done:
                    agent.finish_path(0)
                agent.train()
                b = 0

        if epoch % 1000 == 0:
            print("\nEpoch: %s Rewards mean:%s" % (epoch, np.mean(rewards)))

    for epoch in range(10):
        import time
        print("=========================TEST=================================")
        done = False
        state = grid.gen_state()

        while not done:
            time.sleep(1)
            mu, pi, logpi = agent.step([state])

            y = max(0, min(6, int(round((pi[0][0]+1.) / 2*6))))
            x = max(0, min(6, int(round((pi[0][1]+1.) / 2*6))))

            n_state, _, done = grid.step(y, x)
            grid.display()
            state = n_state
        print("reward=>", reward)

if __name__ == '__main__':
    main()

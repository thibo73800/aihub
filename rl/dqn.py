from collections import deque, namedtuple
from gym.wrappers import Monitor
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import itertools
import random
import gym
import os
import sys

VALID_ACTIONS = [0, 1, 2, 3]


from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']
get_available_gpus()


class StateProcessor():

    def __init__(self):
      with tf.variable_scope("process"):
        self.input_state = tf.placeholder(shape=[210, 160, 3], dtype=tf.uint8, name="input_process")
        self.output = tf.image.rgb_to_grayscale(self.input_state)
        self.output = tf.image.crop_to_bounding_box(self.output, 34, 0, 160, 160)
        self.output = tf.image.resize_images(
            self.output, [84, 84], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        self.output = tf.squeeze(self.output)

    def process(self, sess, state):
        return sess.run(self.output, { self.input_state: state })

import gym
import time

env = gym.make('Breakout-v0')
env.reset()

for _ in range(1000):
    observation, reward, done, info = env.step(env.action_space.sample())
    if done:
        break

class DQN():

    def __init__(self, scope):
      self.scope = scope
      with tf.variable_scope(self.scope):
        self._build_model()

    def _build_model(self):
        # 4 Last frames of the game
        self.X_pl = tf.placeholder(shape=[None, 84, 84, 4], dtype=tf.uint8, name="X")
        # The TD target value
        self.y_pl = tf.placeholder(shape=[None], dtype=tf.float32, name="y")
        # Integer id of which action was selected
        self.actions_pl = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")

        # Rescale the image
        X = tf.to_float(self.X_pl) / 255.0
        # Get the batch size
        batch_size = tf.shape(self.X_pl)[0]

        # Three convolutional layers
        conv1 = tf.layers.conv2d(X, 32, 8, 4, activation=tf.nn.relu)
        conv2 = tf.layers.conv2d(conv1, 64, 4, 2, activation=tf.nn.relu)
        conv3 = tf.layers.conv2d(conv2, 64, 3, 1, activation=tf.nn.relu)

        # Fully connected layers
        flattened = tf.contrib.layers.flatten(conv3)
        fc1 = tf.layers.dense(flattened, 512, activation=tf.nn.relu)
        self.predictions = tf.layers.dense(fc1, len(VALID_ACTIONS))
        tf.identity(self.predictions, name="predictions")

        # Get the predictions for the chosen actions only
        gather_indices = tf.range(batch_size) * tf.shape(self.predictions)[1] + self.actions_pl
        self.action_predictions = tf.gather(tf.reshape(self.predictions, [-1]), gather_indices)

        # Calculate the loss
        self.losses = tf.squared_difference(self.y_pl, self.action_predictions)
        self.loss = tf.reduce_mean(self.losses)

        # Optimizer Parameters from original paper
        self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
        self.train_op = self.optimizer.minimize(self.loss)

    def predict(self, sess, s):
        return sess.run(self.predictions, { self.X_pl: s })

    def update(self, sess, s, a, y):
        feed_dict = { self.X_pl: s, self.y_pl: y, self.actions_pl: a }
        ops = [self.train_op, self.loss]
        _, loss = sess.run(ops, feed_dict)
        return loss

def copy_model_parameters(sess, estimator1, estimator2):
    e1_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator1.scope)]
    e1_params = sorted(e1_params, key=lambda v: v.name)
    e2_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator2.scope)]
    e2_params = sorted(e2_params, key=lambda v: v.name)

    update_ops = []
    for e1_v, e2_v in zip(e1_params, e2_params):
        op = e2_v.assign(e1_v)
        update_ops.append(op)

    sess.run(update_ops)

tf.reset_default_graph()


# DQN
dqn = DQN(scope="dqn")
# DQN target
target_dqn = DQN(scope="target_dqn")


# State processor
state_processor = StateProcessor()

num_episodes = 10000

replay_memory_size = 250000
replay_memory_init_size = 50000

update_target_estimator_every = 10000

epsilon_start = 1.0
epsilon_end = 0.1


epsilon_decay_steps = 500000
discount_factor = 0.99
batch_size = 32

def make_epsilon_greedy_policy(estimator, nA):
    def policy_fn(sess, observation, epsilon):
        A = np.ones(nA, dtype=float) * epsilon / nA
        q_values = estimator.predict(sess, np.expand_dims(observation, 0))[0]
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn

#saver = tf.train.Saver()
start_i_episode = 0
opti_step = -1

# The replay memory
replay_memory = []



with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])


    # Used to save the model
    checkpoint_dir = os.path.join("./", "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, "model")

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    saver = tf.train.Saver()
    # Load a previous checkpoint if we find one
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)

    #  Epsilon decay
    epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)

    # Policy
    policy = make_epsilon_greedy_policy(dqn, len(VALID_ACTIONS))

    epi_reward = []
    best_epi_reward = 0

    for i_episode in range(start_i_episode, num_episodes):
        # Reset the environment
        state = env.reset()
        state = state_processor.process(sess, state)
        state = np.stack([state] * 4, axis=2)
        loss = None
        done = False
        r_sum = 0
        mean_epi_reward = np.mean(epi_reward)
        if best_epi_reward < mean_epi_reward:
            best_epi_reward = mean_epi_reward
            saver.save(tf.get_default_session(), checkpoint_path)

        len_replay_memory = len(replay_memory)
        while not done:
            # Get the epsilon for this step
            epsilon = epsilons[min(opti_step+1, epsilon_decay_steps-1)]


            # Update the target network
            if opti_step % update_target_estimator_every == 0:
                copy_model_parameters(sess, dqn, target_dqn)

            print("\r Epsilon ({}) ReplayMemorySize : ({}) rSum: ({}) best_epi_reward: ({}) OptiStep ({}) @ Episode {}/{}, loss: {}".format(epsilon, len_replay_memory, mean_epi_reward, best_epi_reward, opti_step, i_episode + 1, num_episodes, loss), end="")
            sys.stdout.flush()


            #  Select an action with eps-greedy
            action_probs = policy(sess, state, epsilon)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

            # Step in the env with this action
            next_state, reward, done, _ = env.step(VALID_ACTIONS[action])
            r_sum += reward

            # Add this action to the stack of images
            next_state = state_processor.process(sess, next_state)
            next_state = np.append(state[:,:,1:], np.expand_dims(next_state, 2), axis=2)

            # If our replay memory is full, pop the first element
            if len(replay_memory) == replay_memory_size:
                replay_memory.pop(0)


            # Save transition to replay memory
            replay_memory.append(Transition(state, action, reward, next_state, done))

            if len_replay_memory > replay_memory_init_size:
                # Sample a minibatch from the replay memory
                samples = random.sample(replay_memory, batch_size)
                states_batch, action_batch, reward_batch, next_states_batch, done_batch = map(np.array, zip(*samples))

                # We compute the next q value with
                q_values_next_target = target_dqn.predict(sess, next_states_batch)
                t_best_actions = np.argmax(q_values_next_target, axis=1)
                targets_batch = reward_batch + np.invert(done_batch).astype(np.float32) * discount_factor * q_values_next_target[np.arange(batch_size), t_best_actions]

                # Perform gradient descent update
                states_batch = np.array(states_batch)
                loss = dqn.update(sess, states_batch, action_batch, targets_batch)

                opti_step += 1

            state = next_state
            if done:
              break

        epi_reward.append(r_sum)
        if len(epi_reward) > 100:
            epi_reward = epi_reward[1:]

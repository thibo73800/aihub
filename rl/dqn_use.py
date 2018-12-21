import tensorflow as tf
import numpy as np
import gym
import time
import os

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

tf.reset_default_graph()

checkpoint = os.path.join("./checkpoints", "model")
new_saver = tf.train.import_meta_graph(checkpoint + '.meta')

with tf.Session() as sess:
  # Restore variables from disk.


  new_saver.restore(sess, checkpoint)


  input_data = tf.get_default_graph().get_tensor_by_name('dqn/X:0')
  input_process = tf.get_default_graph().get_tensor_by_name('process/input_process:0')
  probs = tf.get_default_graph().get_tensor_by_name('dqn/predictions:0')

  state_processor = StateProcessor()

  env = gym.make('Breakout-v0')
  env.reset()

  state = env.reset()
  state = state_processor.process(sess, state)
  state = np.stack([state] * 4, axis=2)

  for _ in range(1000):
      import time
      time.sleep(0.05)
      env.render()
      p = sess.run(probs, feed_dict={input_data: [state]})[0]
      action = np.argmax(p)
      next_state, reward, done, info = env.step(action)

      # Add this action to the stack of images
      next_state = state_processor.process(sess, next_state)
      next_state = np.append(state[:,:,1:], np.expand_dims(next_state, 2), axis=2)

      if reward != 0:
          print("Reward", reward)

      state = next_state

      if done:
          break

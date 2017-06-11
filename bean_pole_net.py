# Copyright 2017 Kirby Banman. All Rights Reserved.
#
# Adapted from an original work located at
#
#     https://github.com/tensorflow/tensorflow/blob/3e1676e40aace360440886d823c53cc63a98ace1/tensorflow/examples/tutorials/mnist/mnist_deep.py
#
# by the TensorFlow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import argparse
import time
import sys
import os

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import numpy as np


# pylint: disable=W0311,C0103


FLAGS = None


def bean_pole_net(x_in):
  """Construct and return an MNIST beanpole net with dropout keep probability as a tuple."""
  # Reshape to use within a convolutional neural net.
  # Last dimension is for "features" - there is only one here, since images are
  # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
  x_image = tf.reshape(x_in, [-1, 28, 28, 1])

  tf.summary.image("input image", x_image, collections=["bean_pole_images"])

  # Bean pole layers
  bean_out = bean_pole_layers(x_image, FLAGS.pole_depth, FLAGS.max_skip_depth)

  return bean_out


def bean_pole_layers(x_in, layer_count, max_skip_depth):
  """Construct and return bean pole layers with the input specified."""
  layers = [x_in]
  while len(layers) <= layer_count:
    print("Constructing bean pole layer " + str(len(layers)))
    input_skip_distance = 1

    print("Adding layer " + str(len(layers) - input_skip_distance) + " to input")
    layer_input = hidden_layer(layers[len(layers) - input_skip_distance], 1)
    input_skip_distance += 1
    while input_skip_distance <= max_skip_depth and len(layers) - input_skip_distance >= 0:
      print("Adding layer " + str(len(layers) - input_skip_distance) + " to input")
      layer_input += hidden_layer(layers[len(layers) - input_skip_distance], 1)
      input_skip_distance += 1

    layer = hidden_layer(layer_input, 5)
    tf.summary.image("bean pole image " + str(len(layers)), layer, collections=["bean_pole_images"])

    layers.append(layer)

  return layers[-1]


def hidden_layer(x_in, convolution_size):
  """relu convolution with bias"""
  W_conv = weight_variable([convolution_size, convolution_size, 1, 1])
  b_conv = bias_variable([1])
  h_conv = tf.nn.relu(conv2d(x_in, W_conv) + b_conv)

  return h_conv


def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def imagify_batch(batch):
  """
  maps digits with curly bits to 28x28 all ones, zeros otherwise.
  3, 6, 8, 9, 0
  """
  return [vec_to_image(vec) for vec in batch]


def vec_to_image(vec):
  """Converts an MNIST label vector to a greyscale image."""
  if vec[2] > 0.5 or \
     vec[5] > 0.5 or \
     vec[7] > 0.5 or \
     vec[8] > 0.5 or \
     vec[9] > 0.5:
    return np.ones([28, 28, 1], dtype=np.float32)
  else:
    return np.zeros([28, 28, 1], dtype=np.float32)


def ensure_dir(directory):
  """Ensures the existance of a directory."""
  if not os.path.exists(directory):
    os.makedirs(directory)


def main(_):
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 28, 28, 1])

  # Build the graph for the deep net
  y_conv = bean_pole_net(x)

  mean_squared_error = tf.reduce_mean(tf.squared_difference(y_, y_conv))
  train_step = tf.train.AdamOptimizer(1e-4).minimize(mean_squared_error)

  tf.summary.scalar('mean squared error', mean_squared_error, collections=['train_stats'])
  merged_train_stats = tf.summary.merge_all(key='train_stats')
  merged_bean_pole_images = tf.summary.merge_all(key='bean_pole_images')

  saver = tf.train.Saver()
  with tf.Session() as sess:

    ensure_dir(os.path.join(FLAGS.log_dir, FLAGS.run_name, "train"))
    ensure_dir(os.path.join(FLAGS.train_dir, FLAGS.run_name))

    train_writer = tf.summary.FileWriter(os.path.join(FLAGS.log_dir, FLAGS.run_name), sess.graph)
    sess.run(tf.global_variables_initializer())
    for i in range(20000):
      batch = mnist.train.next_batch(FLAGS.batch_size)
      batch_targets = imagify_batch(batch[1])

      if i % 100 == 0:
        train_error = mean_squared_error.eval(feed_dict={
            x: batch[0], y_: batch_targets})
        print('step %d, training error %g' % (i, train_error))

      if i % 1000 == 0:
        saver.save(sess, os.path.join(FLAGS.train_dir, FLAGS.run_name, "model.ckpt"), global_step=i)

      # Train the network, recording stats every tenth step
      if i % 1000 == 0:
        summary, _ = sess.run( \
          [merged_bean_pole_images, train_step], \
          feed_dict={x: batch[0], y_: batch_targets} \
        )
        train_writer.add_summary(summary, i)
      elif i % 10 == 0:
        summary, _ = sess.run( \
          [merged_train_stats, train_step], \
          feed_dict={x: batch[0], y_: batch_targets} \
        )
        train_writer.add_summary(summary, i)
      else:
        sess.run(train_step, feed_dict={x: batch[0], y_: batch_targets})

    print('test error %g' % mean_squared_error.eval(feed_dict={
        x: mnist.test.images, y_: imagify_batch(mnist.test.labels)}))

if __name__ == '__main__':
  dir_path = os.path.dirname(os.path.realpath(__file__))

  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default=os.path.join(dir_path, 'input_data'),
                      help='Directory for storing input data')
  parser.add_argument('--train_dir', type=str,
                      default=os.path.join(dir_path, 'checkpoints'),
                      help='Directory for storing training checkpoints')
  parser.add_argument('--log_dir', type=str,
                      default=os.path.join(dir_path, 'log'),
                      help='Directory for storing log output')
  parser.add_argument('--pole_depth', type=int,
                      default=10,
                      help='Number of bean pole layers to use between input and output')
  parser.add_argument('--max_skip_depth', type=int,
                      default=3,
                      help='Max number of bean pole layers to skip')
  parser.add_argument('--batch_size', type=int,
                      default=50,
                      help='Training minibatch size')
  parser.add_argument('--run_name', type=str,
                      default=str(int(time.time())),
                      help='Name for the run')
  FLAGS, unparsed = parser.parse_known_args()

  print(FLAGS)
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

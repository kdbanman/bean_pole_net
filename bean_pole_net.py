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
RUN_NAME = None

LAYERS = None


def construct_bean_pole_net(x_in):
  """
  Construct and return an MNIST beanpole net with dropout keep probability as a tuple.
  MODIFIES GLOBAL LAYERS.
  """
  global LAYERS

  # Reshape to use within a convolutional neural net.
  # Last dimension is for "features" - there is only one here, since images are
  # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
  x_image = tf.reshape(x_in, [-1, 28, 28, 1])

  tf.summary.image("1_1_input_image", x_image, collections=["bean_pole_images"], max_outputs=FLAGS.sample_images)

  add_initial_bean_pole_layers(x_image, FLAGS.pole_depth, FLAGS.max_skip_depth)
  bean_out = LAYERS[-1]
  tf.add_to_collection("get_output", bean_out)

  return bean_out


def create_bean_pole_layer(existing_graph, max_skip_depth):
  """
  Add a layer to an existing bean pole net.
  """
  layer_number = len(LAYERS)
  print("Adding bean pole layer " + str(layer_number))

  # Add the previous layer to the new layer's input - this is required to keep the network connected
  print("Adding layer " + str(layer_number - 1) + " to input")
  layer_input = LAYERS[layer_number - 1]

  # Add layers from earlier in the net, up to the max skip depth specified
  input_skip_distance = 1
  while input_skip_distance < max_skip_depth and layer_number - input_skip_distance >= 0:
    skip_layer_index = layer_number - input_skip_distance - 1
    print("Adding layer " + str(skip_layer_index) + " to input")

    # Add the layer with a 1x1 convolution window to weight the entire image with a single parameter.
    layer_input += hidden_layer(LAYERS[skip_layer_index], 1)
    input_skip_distance += 1

  new_output = bean_pole_module(existing_graph, FLAGS.fan_out_channels, FLAGS.convolution_size, "module_layer_" + str(layer_number))

  tf.summary.image("2_bean_pole_image_" + str(layer_number), new_output, collections=["bean_pole_images"], max_outputs=FLAGS.sample_images)
  tf.add_to_collection("get_layer_images", new_output)

  return new_output


def add_initial_bean_pole_layers(x_in, layer_count, max_skip_depth):
  """
  Construct and return bean pole layers with the input specified.
  Using max_skip_depth == 0 means only connect immediately previous layers.
  MODIFIES GLOBAL LAYERS.
  """
  global LAYERS

  tf.add_to_collection("get_layer_images", x_in)
  LAYERS = [x_in]
  while len(LAYERS) <= layer_count and len(LAYERS) < FLAGS.initial_depth:
    layer = create_bean_pole_layer(LAYERS[-1], max_skip_depth)
    LAYERS.append(layer)

  return LAYERS[-1]


def bean_pole_module(x_in, intermediate_channels=5, convolution_size=5, name="bean_pole_module"):
  h_conv = hidden_layer(x_in, convolution_size, 1, intermediate_channels, name + "_1_fan_out")

  output_conv = hidden_layer(h_conv, convolution_size, intermediate_channels, 1, name + "_2_fan_in")

  return output_conv

def hidden_layer(x_in, convolution_size, input_feature_count=1, output_feature_count=1, name="hidden_layer"):
  """relu convolution with bias"""
  W_conv = weight_variable([convolution_size, convolution_size, input_feature_count, output_feature_count])
  b_conv = bias_variable([1])
  h_conv = tf.nn.relu(conv2d(x_in, W_conv) + b_conv)

  if convolution_size > 1:

    # 5x5 3 channel fan OUT W_conv has shape 5,5,1,3.  Need 3,5,5,1 to get an image per filter.
    # 5x5 3 channel fan IN W_conv has shape 5,5,3,1.  Need 3,5,5,1 to get an image per filter.
    feature_maps = tf.stack(tf.split(W_conv, output_feature_count, 3))
    feature_map_image_count = max(input_feature_count, output_feature_count)
    feature_maps = tf.reshape(feature_maps, [feature_map_image_count, convolution_size, convolution_size, 1])
    tf.summary.image("4_" + name + "_weights", feature_maps, collections=["bean_pole_images"], max_outputs=feature_map_image_count)

    activations = tf.split(h_conv, output_feature_count, 3)
    for i in range(0, output_feature_count):
      tf.summary.image("3_" + name + "_activation_" + str(i), activations[i], collections=["bean_pole_images"], max_outputs=FLAGS.sample_images)

  return h_conv


def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")


def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=FLAGS.initialization_stddev)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def imagify_batch(batch):
  """
  maps even digits to 28x28 all ones, zeros otherwise.
  """
  return [vec_to_image(vec) for vec in batch]


def vec_to_image(vec):
  """Converts an MNIST label vector to a greyscale image."""
  if vec[2] > 0.5 or \
     vec[4] > 0.5 or \
     vec[6] > 0.5 or \
     vec[8] > 0.5 or \
     vec[0] > 0.5:
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
  x = tf.placeholder(tf.float32, [None, 784], name="input_placeholder")

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 28, 28, 1])

  # Build the graph for the deep net
  y_conv = construct_bean_pole_net(x)
  tf.summary.image("1_2_output_image", y_conv, collections=["bean_pole_images"], max_outputs=FLAGS.sample_images)

  mean_squared_error = tf.reduce_mean(tf.squared_difference(y_, y_conv))
  train_step = tf.train.AdamOptimizer(1e-4).minimize(mean_squared_error)

  tf.summary.scalar("mean_squared_train_error", mean_squared_error, collections=["train_stats"])
  merged_train_stats = tf.summary.merge_all(key="train_stats")
  merged_bean_pole_images = tf.summary.merge_all(key="bean_pole_images")
  histograms = tf.summary.merge_all(key="histograms")

  saver = tf.train.Saver()
  with tf.Session() as sess:

    ensure_dir(os.path.join(FLAGS.log_dir, RUN_NAME, "train"))
    ensure_dir(os.path.join(FLAGS.train_dir, RUN_NAME))

    train_writer = tf.summary.FileWriter(os.path.join(FLAGS.log_dir, RUN_NAME), sess.graph)
    sess.run(tf.global_variables_initializer())
    i = 0
    while True:
      batch = mnist.train.next_batch(FLAGS.batch_size)
      batch_targets = imagify_batch(batch[1])

      if i % 100 == 0:
        train_error = mean_squared_error.eval(feed_dict={
            x: batch[0], y_: batch_targets})

        if train_error < FLAGS.layer_add_threshold and len(LAYERS) <= FLAGS.pole_depth:
            LAYERS.append(create_bean_pole_layer(LAYERS[-1], FLAGS.max_skip_depth))

        print("step %d, training error %g" % (i, train_error))

      if i % 1000 == 0:
        saver.save(sess, os.path.join(FLAGS.train_dir, RUN_NAME, "model.ckpt"), global_step=i)
        test_error = mean_squared_error.eval(feed_dict={x: mnist.test.images, y_: imagify_batch(mnist.test.labels)})
        print("step %d, test error %g" % (i, test_error))
        tf.summary.scalar("mean_squared_test_error", test_error, collections=["train_stats"])

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

      i += 1

if __name__ == "__main__":
  dir_path = os.path.dirname(os.path.realpath(__file__))

  parser = argparse.ArgumentParser()
  parser.add_argument("--data_dir", type=str,
                      default=os.path.join(dir_path, "input_data"),
                      help="Directory for storing input data")
  parser.add_argument("--train_dir", type=str,
                      default=os.path.join(dir_path, "checkpoints"),
                      help="Directory for storing training checkpoints")
  parser.add_argument("--log_dir", type=str,
                      default=os.path.join(dir_path, "log"),
                      help="Directory for storing log output")
  parser.add_argument("--pole_depth", type=int,
                      default=10,
                      help="Number of bean pole layers to use between input and output")
  parser.add_argument("--initial_depth", type=int,
                      default=8,
                      help="Number of bean pole layers to train before adding more")
  parser.add_argument("--layer_add_threshold", type=float,
                      default=0.15,
                      help="Training loss threshold at which to add layers (up to pole_depth)")
  parser.add_argument("--max_skip_depth", type=int,
                      default=1,
                      help="Max number of bean pole layers to skip")
  parser.add_argument("--fan_out_channels", type=int,
                      default=3,
                      help="Number of feature maps to use in each bean pole layer module")
  parser.add_argument("--convolution_size", type=int,
                      default=5,
                      help="Convolution window size to use in bean pole modules.")
  parser.add_argument("--initialization_stddev", type=float,
                      default=0.1,
                      help="Standard deviation to use when initializing the network weights.")
  parser.add_argument("--batch_size", type=int,
                      default=50,
                      help="Training minibatch size")
  parser.add_argument("--sample_images", type=int,
                      default=5,
                      help="Number of images to sample during training")
  FLAGS, unparsed = parser.parse_known_args()

  RUN_NAME = str(int(time.time())) + "_" + str(FLAGS.pole_depth) + "Hx" + str(FLAGS.fan_out_channels) + "Wx" + str(FLAGS.convolution_size) + "Cx" + str(FLAGS.max_skip_depth) + "S"
  print(FLAGS)
  print(RUN_NAME)

  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

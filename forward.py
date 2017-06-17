import argparse
import glob
import os

glob.glob("/path/to/directory/*/")

import imageio
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

if __name__ == '__main__':
  dir_path = os.path.dirname(os.path.realpath(__file__))

  parser = argparse.ArgumentParser()
  parser.add_argument('--mnist_dir', type=str,
                      default=os.path.join(dir_path, 'input_data'),
                      help='Directory for storing mnist data')
  parser.add_argument('--model_checkpoint', type=str,
                      default=os.path.join(dir_path, 'checkpoints'),
                      help='Meta file for graph to load')

  FLAGS, unparsed = parser.parse_known_args()

  meta_file = sorted(glob.glob(os.path.join(sorted(glob.glob(os.path.join(FLAGS.model_checkpoint, "*")), key=os.path.getmtime)[-1], "*.meta")), key=os.path.getmtime)[-1]
  model_location = meta_file[:-5]
  print(meta_file)
  print(model_location)

  with tf.Session() as sess:
    importer = tf.train.import_meta_graph(meta_file)
    importer.restore(sess, model_location)
    get_output = tf.get_collection("get_output")
    get_layer_images = tf.get_collection("get_layer_images")

    mnist = input_data.read_data_sets(FLAGS.mnist_dir)
    input_image = mnist.train.next_batch(1, shuffle=True)[0]
    input_placeholder = sess.graph.get_tensor_by_name("input_placeholder:0")

    output = sess.run(get_output, feed_dict={input_placeholder: input_image})
    layer_output = sess.run(get_layer_images, feed_dict={input_placeholder: input_image})

    output_image = tf.reshape(output[0], (28, 28)).eval()
    layer_images = [tf.reshape(layer, (28, 28)).eval() for layer in layer_output]

    imageio.imwrite("output_image.png", output_image)
    imageio.mimwrite("layer_images.gif", layer_images, duration=0.5)
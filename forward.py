import argparse
import glob
import os

import imageio
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

FLAGS = None


def get_input_image(filename):
  image = imageio.imread(filename)
  if image.shape == (28, 28, 3):
    return [image[:,:,0].reshape((784))]
  elif image.shape == (28, 28, 1):
    return [image.reshape((784))]
  else:
    raise Exception("Incompatible input image shape: " + str(image.shape))


def get_outputs(sess, input_image):
  input_placeholder = sess.graph.get_tensor_by_name("input_placeholder:0")

  get_output = tf.get_collection("get_output")
  get_layer_images = tf.get_collection("get_layer_images")
  
  output = sess.run(get_output, feed_dict={input_placeholder: input_image})
  layer_output = sess.run(get_layer_images, feed_dict={input_placeholder: input_image})

  output_image = tf.reshape(output[0], (28, 28)).eval()
  layer_images = [tf.reshape(layer, (28, 28)).eval() for layer in layer_output]

  return output_image, layer_images


def write_outputs(input_filename, output_image, layer_images):
  basename = os.path.basename(filename)
  input_name = os.path.splitext(basename)[0]

  imageio.imwrite(os.path.join(FLAGS.input_dir, input_name + "_output_image.png"), output_image)
  imageio.mimwrite(os.path.join(FLAGS.input_dir, input_name + "layer_images.gif"), layer_images, duration=0.5)


if __name__ == '__main__':
  dir_path = os.path.dirname(os.path.realpath(__file__))

  parser = argparse.ArgumentParser()
  parser.add_argument('--model_checkpoint', type=str,
                      default=sorted(glob.glob(os.path.join(sorted(glob.glob(os.path.join(os.path.join(dir_path, 'checkpoints'), "*")), key=os.path.getmtime)[-1], "*.meta")), key=os.path.getmtime)[-1],
                      help='Meta file for graph to load')
  parser.add_argument('--input_dir', type=str,
                      default=None,
                      help='Directory to load .png images from.')

  FLAGS, unparsed = parser.parse_known_args()

  meta_file = FLAGS.model_checkpoint
  model_location = meta_file[:-5]

  with tf.Session() as sess:
    importer = tf.train.import_meta_graph(meta_file)
    importer.restore(sess, model_location)

    for filename in sorted(glob.glob(os.path.join(FLAGS.input_dir, "*.png"))):
      input_image = get_input_image(filename)

      output_image, layer_images = get_outputs(sess, input_image)

      write_outputs(filename, output_image, layer_images)
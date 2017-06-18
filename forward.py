import argparse
import glob
import os

import imageio
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

FLAGS = None

HTML_TEMPLATE = """
  <html>
  <head>
    <style>
    body {
      width: auto;
      overflow-x: scroll;
      white-space: nowrap;
    }
    .row {
      margin-bottom: 15px;
    }
    .padRightSmall {
      padding-right: 5px;
    }
    .padRightLarge {
      padding-right: 20px;
    }
    code {
      font-size: 14px;
      vertical-align: top;
    }
    </style>
  </head>
  <body>
  $ROWS$
  </body>
  </html>
"""

ROW_TEMPLATE = """
  <div class="row">
    <code>In:</code><img class="padRightSmall" src="$DIR_NAME$/$INPUT_IMAGE_FILENAME$"/>
    <code>Out:</code><img class="padRightLarge" src="$DIR_NAME$/$OUTPUT_IMAGE_FILENAME$"/>
    <code>Animated:</code><img class="padRightLarge" src="$DIR_NAME$/$LAYER_VIDEO_FILENAME$"/>
    <code>Frames:</code>$LAYER_IMAGES$
  </div>
"""

LAYER_IMAGE_TEMPLATE = """
  <img class="padRightSmall" src="$DIR_NAME$/$LAYER_IMAGE_FILENAME$"/>
"""


def get_input_image(filename):
  image = imageio.imread(filename)
  if image.shape == (28, 28, 3):
    return [image[:,:,0].reshape((784))]
  elif image.shape == (28, 28, 1) or image.shape == (28, 28):
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


def write_result_images(input_filepath, output_image, layer_images):
  input_image_filename = os.path.basename(input_filepath)
  input_name_no_extension = os.path.splitext(input_image_filename)[0]

  output_image_filepath = os.path.join(FLAGS.input_dir, input_name_no_extension + "_output_image.png")
  layer_video_filepath = os.path.join(FLAGS.input_dir, input_name_no_extension + "_layer_images.gif")

  output_image_filename = os.path.basename(output_image_filepath)
  layer_video_filename = os.path.basename(layer_video_filepath)

  imageio.imwrite(output_image_filepath, output_image)
  imageio.mimwrite(layer_video_filepath, layer_images, duration=0.5)

  i = 0
  layer_image_filenames = []
  for layer_image in layer_images:
    layer_image_filename = os.path.join(FLAGS.input_dir, input_name_no_extension + "_layer_image_" + str(i) + ".png")
    imageio.imwrite(layer_image_filename, layer_image)

    layer_image_filenames.append(os.path.basename(layer_image_filename))
    i += 1

  return {
    "input_image_filename": input_image_filename,
    "layer_video_filename": layer_video_filename,
    "output_image_filename": output_image_filename,
    "layer_image_filenames": layer_image_filenames
  }


def write_html_summary(result_filenames):
  rows = []
  for result_row_filenames in result_filenames:
    input_image_filename = result_row_filenames["input_image_filename"]
    layer_video_filename = result_row_filenames["layer_video_filename"]
    output_image_filename = result_row_filenames["output_image_filename"]
    layer_image_filenames = result_row_filenames["layer_image_filenames"]

    layer_image_tags = "".join([LAYER_IMAGE_TEMPLATE.replace("$LAYER_IMAGE_FILENAME$", layer_image_filename) for layer_image_filename in layer_image_filenames])
    row = ROW_TEMPLATE.replace("$INPUT_IMAGE_FILENAME$", input_image_filename).replace("$LAYER_VIDEO_FILENAME$", layer_video_filename).replace("$OUTPUT_IMAGE_FILENAME$", output_image_filename).replace("$LAYER_IMAGES$", layer_image_tags)

    rows.append(row)

  input_dir_name = os.path.basename(FLAGS.input_dir)
  html = HTML_TEMPLATE.replace("$ROWS$", "".join(rows)).replace("$DIR_NAME$", input_dir_name)

  with open(os.path.join(os.path.dirname(FLAGS.input_dir), input_dir_name + " - Results.html"), "w") as f:
    f.write(html)


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

    result_filenames = []
    for input_file_path in sorted(glob.glob(os.path.join(FLAGS.input_dir, "*.png"))):
      input_image = get_input_image(input_file_path)

      output_image, layer_images = get_outputs(sess, input_image)

      result_filenames.append(write_result_images(input_file_path, output_image, layer_images))

    write_html_summary(result_filenames)
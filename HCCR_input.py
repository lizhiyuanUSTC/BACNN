from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import HCCR_FLAGS

FLAGS = HCCR_FLAGS.FLAGS

def read_HCCR(filename_queue):
  """Reads and parses examples from HCCR data files.

  Recommendation: if you want N-way read parallelism, call this function
  N times.  This will give you N independent Readers reading different
  files & positions within those files, which will give better mixing of
  examples.

  Args:
    filename_queue: A queue of strings with the filenames to read from.

  Returns:
    An object representing a single example, with the following fields:
      img: a [height, width, depth] uint8 Tensor with the image data
      label: an int32 Tensor with the label.    
  """
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  features = tf.parse_single_example(serialized_example,
                              features={'label': tf.FixedLenFeature([], tf.int64),
                                        'img_raw': tf.FixedLenFeature([], tf.string)})
  img = tf.decode_raw(features['img_raw'], tf.uint8)
  img = tf.reshape(img, [FLAGS.IMAGE_SIZE, FLAGS.IMAGE_SIZE, 1])
  label = tf.cast(features['label'], tf.int32)
  return img, label
  
def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
  """Construct a queued batch of images and labels.

  Args:
    image: 3-D Tensor of [height, width, 1] of type.float32.
    label: 1-D Tensor of type.int32
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.
    batch_size: Number of images per batch.
    shuffle: boolean indicating whether to use a shuffling queue.

  Returns:
    images: Images. 4D tensor of [batch_size, height, width, 1] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  # Create a queue that shuffles the examples, and then
  # read 'batch_size' images + labels from the example queue.

  if shuffle:
    images, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        num_threads=FLAGS.num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)
  else:
    images, label_batch = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=1,
        capacity=min_queue_examples + 3 * batch_size)

  # Display the training images in the visualizer.
  tf.summary.image('images', images)

  return images, tf.reshape(label_batch, [batch_size])    
  
def distorted_inputs(data_dir, batch_size):
  """Construct distorted input for CIFAR training using the Reader ops.

  Args:
    data_dir: Path to the HCCR data directory.
    batch_size: Number of images per batch.

  Returns:
    images: Images. 4D tensor of [batch_size, FLAGS.IMAGE_SIZE, FLAGS.IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """

  # Create a queue that produces the filenames to read.
  filename_queue = tf.train.string_input_producer(data_dir)

  # Read examples from files in the filename queue.
  image, label = read_HCCR(filename_queue)
  float_image = tf.cast(image, tf.float32)

  # Image processing for training the network. Note the many random
  # distortions applied to the image.

  # Because these operations are not commutative, consider randomizing
  # the order their operation.
  #distorted_image = tf.image.random_brightness(distorted_image,
  #                                             max_delta=63)
  #distorted_image = tf.image.random_contrast(distorted_image,
  #                                           lower=0.2, upper=1.8)

  # Subtract off the mean and divide by the variance of the pixels.
  #float_image = tf.image.per_image_standardization(distorted_image)
  float_image = tf.image.resize_images(float_image, [FLAGS.IMAGE_SIZE, FLAGS.IMAGE_SIZE])
  # Set the shapes of tensors.
  float_image.set_shape([FLAGS.IMAGE_SIZE, FLAGS.IMAGE_SIZE, 1])

  # Ensure that the random shuffling has good mixing properties.
  min_queue_examples = FLAGS.batch_size * 1000
  print ('Filling queue with %d HCCR images before starting to train. '
         'This will take a few minutes.' % min_queue_examples)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(float_image, label,
                                         min_queue_examples, batch_size,
                                         shuffle=True)
  
def inputs(data_dir, batch_size):
  """Construct input for CIFAR evaluation using the Reader ops.

  Args:
    eval_data: bool, indicating if one should use the train or eval data set.
    data_dir: Path to the CIFAR-10 data directory.
    batch_size: Number of images per batch.

  Returns:
    images: Images. 4D tensor of [batch_size, FLAGS.IMAGE_SIZE, FLAGS.IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """

  # Create a queue that produces the filenames to read.
  filename_queue = tf.train.string_input_producer(data_dir)

  # Read examples from files in the filename queue.
  image, label = read_HCCR(filename_queue)
  float_image = tf.cast(image, tf.float32)

  # Image processing for evaluation.

  # Subtract off the mean and divide by the variance of the pixels.
  #float_image = tf.image.per_image_standardization(float_image)

  # Set the shapes of tensors.
  float_image.set_shape([FLAGS.IMAGE_SIZE, FLAGS.IMAGE_SIZE, 1])

  # Ensure that the random shuffling has good mixing properties.
  min_queue_examples = FLAGS.batch_size * 1000

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(float_image, label,
                                         min_queue_examples, batch_size,
                                         shuffle=False)

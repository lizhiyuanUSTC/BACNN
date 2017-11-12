from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time
import os
import importlib

import numpy as np
import tensorflow as tf

import HCCR
import HCCR_FLAGS
import ChineseNet

FLAGS = HCCR_FLAGS.FLAGS

tf.app.flags.DEFINE_string('eval_dir', 'HCCR_eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', '/media/ai/DL_DATA/HCCR/THRESH_OTSU/competition.tfrecords',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 224419,
                            """Number of examples to run.""")
# Basic model parameters.
tf.app.flags.DEFINE_boolean('is_aug', False,
                            """Eval the model using aug data.""")



def eval_once(saver, top_k_op):
  """Run Eval once.

  Args:
    saver: Saver.
    top_k_op: Top K op.
  """
  acc = []
  model_weight_paths = ['model.ckpt-' + str(i) for i in range(FLAGS.save_checkpoint_steps, FLAGS.max_steps+1, FLAGS.save_checkpoint_steps)]
  model_count = 0
  #model_weight_paths = ['model.ckpt-390000']
  for path in model_weight_paths:
    
    with tf.Session() as sess:
      saver.restore(sess, os.path.join('model_weights', path))
      # Start the queue runners.
      coord = tf.train.Coordinator()
      try:
        threads = []
        for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
          threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                           start=True))

        num_iter = int(math.floor(FLAGS.num_examples / FLAGS.eval_batch_size))
        true_count = 0  # Counts the number of correct predictions.
        #total_sample_count = num_iter * FLAGS.eval_batch_size
        last_batch_size = FLAGS.num_examples - num_iter * FLAGS.eval_batch_size
        step = 0
        while step < num_iter + 1 and not coord.should_stop():
          predictions,  = sess.run([top_k_op])
          # print('%s: Model %d, Batch %d, precision @ 1 = %.3f' % (datetime.now(), model_count, step, np.sum(predictions) / FLAGS.eval_batch_size))
          if step == num_iter:
            predictions = predictions[:last_batch_size]
          true_count += np.sum(predictions)
          step += 1

        # Compute precision @ 1.
        precision = true_count / FLAGS.num_examples
        acc.append(precision)
        print('%d / %d' % (true_count, FLAGS.num_examples))
        print('%s: Model %d, precision @ 1 = %.6f' % (datetime.now(), model_count, precision))
        model_count += 1

      except Exception as e:  # pylint: disable=broad-except
        coord.request_stop(e)
  
      coord.request_stop()
      coord.join(threads, stop_grace_period_secs=10)
  return np.array(acc)



def evaluate():
 # FLAGS.batch_szie = 1000
  with tf.Graph().as_default() as g:
    images, labels = HCCR.inputs(FLAGS.eval_data)
    all_labels = labels

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = ChineseNet.inference(images, phase_train=False)
    if FLAGS.use_fp16:
        logits = tf.cast(logits, tf.float32)
    # Calculate predictions.
    top_k_op = tf.nn.in_top_k(logits, labels, 1)

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(0.9999)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)   
    #saver = tf.train.Saver()
    acc = eval_once(saver, top_k_op)
    np.save('competition.npy', acc)

def main(argv=None):  # pylint: disable=unused-argument
  evaluate()


if __name__ == '__main__':
  os.environ['CUDA_VISIBLE_DEVICES'] = '1'
  tf.app.run()

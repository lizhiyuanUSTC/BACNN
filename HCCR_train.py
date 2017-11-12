from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time
import os
import tensorflow as tf
import HCCR
import HCCR_FLAGS
import ChineseNet
import numpy as np

FLAGS = HCCR_FLAGS.FLAGS

def train():
  os.environ['CUDA_VISIBLE_DEVICES'] = '0'
  if not os.path.exists('model_weights'):
    os.mkdir('model_weights')
  history = []
  
  """Train HCCR for a number of steps."""
  with tf.Graph().as_default():
    global_step = tf.contrib.framework.get_or_create_global_step()

    # Get images and labels for HCCR.
    images, labels = HCCR.distorted_inputs()

    # Build a Graph that computes the logits predictions from the
    # inference model.
    
    logits = ChineseNet.inference(images, weight_decay=FLAGS.weight_decay)

    # Calculate loss.
    total_loss, softmax_loss, accuracy = HCCR.loss(logits, labels)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op, learning_rate = HCCR.train(total_loss, global_step)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep = int(FLAGS.max_steps / FLAGS.save_checkpoint_steps) + 5)
    
    gpu_options = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.7)
    with tf.Session(config=tf.ConfigProto(
        log_device_placement=FLAGS.log_device_placement, 
        gpu_options=gpu_options)) as sess:
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        
        for step in range(FLAGS.max_steps):
            start_time = time.time()
            _, lr, _total_loss, loss, acc = sess.run([train_op, learning_rate, total_loss, softmax_loss, accuracy])
            history.append([_total_loss, loss, acc])
            duration = time.time() - start_time
            if step % FLAGS.log_steps == 0:
                num_examples_per_step = FLAGS.batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)
                format_str = ('%s: Iter %d, lr = %.4f, total_loss = %.2f, softmax_loss = %.2f, acc = %.4f (%.1f examples/sec; %.3f '
                        'sec/batch)')
                print (format_str % (datetime.now(), step, lr, _total_loss, loss, acc,
                               examples_per_sec, sec_per_batch))
            if (step+1) % FLAGS.save_checkpoint_steps == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join('model_weights', 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step+1)
        coord.request_stop()
        coord.join(threads)
        np.save('history.npy', np.array(history))

def main(argv=None):  # pylint: disable=unused-argument
  train()


if __name__ == '__main__':
  tf.app.run()

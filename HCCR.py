from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
import HCCR_input
import HCCR_FLAGS

FLAGS = HCCR_FLAGS.FLAGS

def distorted_inputs():
    """Construct distorted input for HCCR training using the Reader ops.

    Returns:
        images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 1] size.
        labels: Labels. 1D tensor of [batch_size] size.

    Raises:
        ValueError: If no data_dir
    """
    if not FLAGS.train_data:
        raise ValueError('Please supply a data_dir')
    images, labels = HCCR_input.distorted_inputs(data_dir=[FLAGS.train_data],
                                                 batch_size=FLAGS.batch_size)
    if FLAGS.use_fp16:
        images = tf.cast(images, tf.float16)
    return images, labels


def inputs(data_dir):
    """Construct input for HCCR evaluation using the Reader ops.

    Args:
        eval_data: bool, indicating if one should use the train or eval data set.

    Returns:
        images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 1] size.
        labels: Labels. 1D tensor of [batch_size] size.

    Raises:
        ValueError: If no data_dir
    """
    if not data_dir:
        raise ValueError('Please supply a data_dir')
    images, labels = HCCR_input.inputs(data_dir=[data_dir],
                                       batch_size=FLAGS.eval_batch_size)
    if FLAGS.use_fp16:
        images = tf.cast(images, tf.float16)
    return images, labels

def loss(logits, labels):
    # Calculate the average cross entropy loss across the batch.
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    predicted_labels = tf.argmax(logits, axis=1)
    accuracy = tf.reduce_mean(tf.to_float(tf.equal(labels, predicted_labels)))
    return tf.add_n([cross_entropy_mean] + regularization_losses, name='total_loss'), cross_entropy_mean, accuracy

def _add_loss_summaries(total_loss):
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    #for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
    #    tf.summary.scalar(l.op.name + ' (raw)', l)
    #    tf.summary.scalar(l.op.name, loss_averages.average(l))

    return loss_averages_op

def train(total_loss, global_step):
    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(FLAGS.INITIAL_LEARNING_RATE,
                                    global_step,
                                    FLAGS.decay_steps,
                                    FLAGS.LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)

    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = _add_loss_summaries(total_loss)

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.MomentumOptimizer(lr, 0.9)
        grads = opt.compute_gradients(total_loss)

    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    # for var in tf.trainable_variables():
    #     tf.summary.histogram(var.op.name, var)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(0.9999, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op, lr

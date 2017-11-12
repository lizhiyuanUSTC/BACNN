from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
import HCCR_FLAGS

FLAGS = HCCR_FLAGS.FLAGS

def round_through(x):
    rounded = tf.round(x)
    return x + tf.stop_gradient(rounded - x)

def _hard_sigmoid(x):
    x = (0.5 * x) + 0.5
    return tf.clip_by_value(x, 0, 1)

def binary_sigmoid(x):
    return round_through(_hard_sigmoid(x))

def binary_tanh(x):
    return 2 * round_through(_hard_sigmoid(x)) - 1


def batch_norm(x):
    return slim.batch_norm(x, decay=0.995, updates_collections=None, variables_collections=[tf.GraphKeys.TRAINABLE_VARIABLES])

def inference(images, phase_train=True, weight_decay=0.0, reuse=None):
    """Build the HCCR model.

    Args:
        images: Images returned from distorted_inputs() or inputs().

    Returns:
        Logits.
    """
    
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        activation_fn=None):
        return ChineseNet(images, weight_decay, is_training=phase_train, reuse=reuse)
    
def ChineseNet(inputs, weight_decay, is_training=True, reuse=None, scope='ChineseNet'):
    inputs = (inputs / 255) * 2 - 1
    print('inputs:', inputs.shape[1:])
    with tf.variable_scope(scope, 'ChineseNet', [inputs], reuse=reuse):
        with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
            with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                                stride=1, padding='SAME'):
                net = slim.conv2d(inputs, 64, 3, padding='SAME', scope='Conv1')
                net = slim.max_pool2d(net, 2, stride=2, padding='SAME')
                net = batch_norm(net)
                net = binary_tanh(net)
                print('Conv1:', net.shape[1:])

                net = slim.conv2d(net, 128, 3, padding='SAME', scope='Conv2_1')
                net = batch_norm(net)
                net = binary_tanh(net)
                print('Conv2_1:', net.shape[1:])

                net = slim.conv2d(net, 128, 3, padding='SAME',scope='Conv2_2')
                net = slim.max_pool2d(net, 2, stride=2, padding='SAME')
                net = batch_norm(net)
                net = binary_tanh(net)
                print('Conv2_2:', net.shape[1:])

                net = slim.conv2d(net, 256, 3, padding='SAME',scope='Conv3_1')
                net = batch_norm(net)
                net = binary_tanh(net)
                print('Conv3_1:', net.shape[1:])

                net = slim.conv2d(net, 256, 3, padding='SAME', scope='Conv3_2')
                net = slim.max_pool2d(net, 2, stride=2, padding='SAME')
                net = batch_norm(net)
                net = binary_tanh(net)
                print('Conv3_2:', net.shape[1:])

                net = slim.conv2d(net, 512, 3, padding='SAME',scope='Conv4_1')
                net = batch_norm(net)
                net = binary_tanh(net)
                print('Conv4_1:', net.shape[1:])

                net = slim.conv2d(net, FLAGS.embedding_size, 3, padding='SAME',scope='Conv4_2')              
                net = tf.reduce_mean(net, [1, 2])
                net = slim.flatten(net)
                print('flatten:', net.shape)
                net = slim.dropout(net, 0.5)
                
                net = slim.fully_connected(net, FLAGS.NUM_CLASSES, scope='Logits')
                


    return net

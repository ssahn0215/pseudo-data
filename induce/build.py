import tensorflow as tf
import numpy as np

import sys
import os

sys.path.extend([os.path.join(sys.path[0],'..')])

from step_dynamic import mala
from models.ensemble_model import build_log_probs
def build(data_loader, model, config):
    print(config)
    """
    Helper Variables
    """
    global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
    global_step_inc = global_step_tensor.assign(global_step_tensor+1)

    tf.add_to_collection('global_step', global_step_tensor)
    tf.add_to_collection('step_ops', global_step_inc)

    """
    Inputs to the network
    """
    with tf.variable_scope('inputs'):
        x, y = data_loader.get_input()
        training = tf.placeholder(tf.bool, name='training_flag')

    tf.add_to_collection('inputs', x)
    tf.add_to_collection('inputs', y)

    """
    Pseudo-inputs to the network
    """
    x_pseudo, y_pseudo = data_loader.get_pseudo_input()
    x_pseudo = tf.Variable(x_pseudo, name='x_pseudo')
    y_pseudo = tf.Variable(y_pseudo, trainable=False, name='y_pseudo')
    tf.add_to_collection('inputs_pseudo', x_pseudo)
    tf.add_to_collection('inputs_pseudo', y_pseudo)

    """
    Network with real input
    """
    _ = model.build_model(x, y, name='ensemble_model')
    weights, logits, log_liks, log_priors, log_probs = _

    """
    Network with pseudo input
    """
    with tf.variable_scope('ensemble_model_pseudo'):
        logits_pseudo = model.build_logits(x_pseudo)
        log_liks_pseudo = model.build_log_liks_from_logits(
            logits_pseudo, y_pseudo)
        log_probs_pseudo = log_liks_pseudo+log_priors

    tf.add_to_collection('losses', tf.reduce_mean(log_probs_pseudo))

    """
    Walk process
    """
    def log_probs_pseudo_fn(_weights):
        return build_log_probs(
            _weights, x_pseudo, y_pseudo, model.layer_dims, model.data_size)

    walk_op, acpt_rate = mala(
        weights,
        log_probs_pseudo_fn,
        config.walk_size,
        config.num_walks_per_step)

    tf.add_to_collection('step_ops', walk_op)
    tf.add_to_collection('acpt_rate', acpt_rate)
    """
    Fisher divergence optimization
    """
    log_liks_grads = tf.gradients(log_liks, [weights])[0]
    log_liks_pseudo_grads = tf.gradients(log_liks_pseudo, [weights])[0]
    fisher = (log_liks_grads-log_liks_pseudo_grads)/config.train_data_size
    fisher = tf.reduce_mean(tf.square(fisher))
    fisher_grad = tf.gradients(fisher, [x_pseudo])[0]
    fisher_opt_op = tf.train.GradientDescentOptimizer(
        learning_rate=config.fisher_learning_rate).minimize(fisher)

    tf.add_to_collection('losses', fisher)
    tf.add_to_collection('step_ops', fisher_opt_op)

    """
    Other statistics
    """
    with tf.variable_scope('log_softmaxs'):
        log_softmaxs = tf.nn.log_softmax(logits)

    with tf.variable_scope('error'):
        avg_log_softmax = tf.reduce_mean(log_softmaxs, axis=0)
        sparse_labels = tf.argmax(y, axis=1, output_type=tf.int32)
        sparse_pred = tf.argmax(avg_log_softmax, axis=1, output_type=tf.int32)
        comp = tf.cast(tf.equal(sparse_labels, sparse_pred), tf.float32)
        error = 100.*(1.-tf.reduce_mean(comp))

    tf.add_to_collection('error', error)

    saver = tf.train.Saver(max_to_keep=config.max_to_keep, save_relative_paths=True)
    tf.add_to_collection('saver', saver)

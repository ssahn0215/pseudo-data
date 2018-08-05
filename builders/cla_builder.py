import tensorflow as tf
import numpy as np

import sys
import os

sys.path.extend([os.path.join(sys.path[0],'..')])

from models.ensemble_model import build_log_probs

class ClassifierBuilder:
    def __init__(self, data_loader, model, config):
        self.data_loader = data_loader
        self.model = model
        self.config = config

    def build(self):
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
            x, y = self.data_loader.get_input()
            training = tf.placeholder(tf.bool, name='training_flag')

        tf.add_to_collection('inputs', x)
        tf.add_to_collection('inputs', y)

        """
        Network with real input
        """
        _ = self.model.build_model(x, y, name='ensemble_model')
        weights, logits, log_liks, log_priors, log_probs = _

        """
        Train process
        """
        def log_probs_fn(_weights):
            return build_log_probs(
                _weights, x, y, self.model.layer_dims, self.model.data_size)

        loss = -tf.reduce_mean(log_probs)/self.config.train_data_size
        walk_op, acpt_rate = mala(
            weights,
            log_probs_fn,
            self.config.walk_size,
            self.config.num_walks_per_step)

        tf.add_to_collection('step_ops', walk_op)
        tf.add_to_collection('losses', loss)
        tf.add_to_collection('acpt_rate', acpt_rate)

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

        saver = tf.train.Saver(max_to_keep=self.config.max_to_keep, save_relative_paths=True)
        tf.add_to_collection('saver', saver)

def mala(thetas, log_probs_fn, lr, nb_walk_per_step):
    avg_acceptance_rate_ = 0.0
    thetas_ = thetas
    log_probs_ = log_probs_fn(thetas_)
    grads_ = tf.gradients(log_probs_, thetas_)[0]
    for step in range(nb_walk_per_step):
        eps = tf.random_normal(thetas_.get_shape())
        proposed_thetas_ = tf.stop_gradient(thetas_+lr*grads_+tf.sqrt(2*lr)*eps)
        proposed_log_probs_ = log_probs_fn(proposed_thetas_)
        proposed_grads_ = tf.gradients(proposed_log_probs_, proposed_thetas_)[0]

        # add rejection step
        log_numer = proposed_log_probs_-0.25/lr*tf.reduce_sum(
            tf.square(thetas_-proposed_thetas_-lr*proposed_grads_), axis=1)
        log_denom = log_probs_-0.5*tf.reduce_sum(tf.square(eps), axis=1)
        acceptance_rate = tf.clip_by_value(tf.exp(log_numer-log_denom), 0.0, 1.0)

        # accept samples and update related quantities
        u = tf.random_uniform(acceptance_rate.get_shape())
        accept = tf.less_equal(u, acceptance_rate)
        thetas_ = tf.where(accept, proposed_thetas_, thetas_)
        log_probs_ = tf.where(accept, proposed_log_probs_, log_probs_)

        avg_acceptance_rate_ += tf.reduce_mean(acceptance_rate)/nb_walk_per_step
        if step < nb_walk_per_step-1:
            grads_ = tf.where(accept, proposed_grads_, grads_)

    return tf.assign(thetas, thetas_), avg_acceptance_rate_

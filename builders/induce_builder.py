import tensorflow as tf
import numpy as np

import sys
import os

sys.path.extend([os.path.join(sys.path[0],'..')])

from models.ensemble_model import build_log_probs

class InducerBuilder:
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
        Pseudo-inputs to the network
        """
        x_pseudo, y_pseudo = self.data_loader.get_pseudo_input()
        x_pseudo = tf.Variable(x_pseudo, name='x_pseudo')
        y_pseudo = tf.Variable(y_pseudo, trainable=False, name='y_pseudo')
        tf.add_to_collection('inputs_pseudo', x_pseudo)
        tf.add_to_collection('inputs_pseudo', y_pseudo)

        """
        Network with real input
        """
        _ = self.model.build_model(x, y, name='ensemble_model')
        weights, logits, log_liks, log_priors, log_probs = _

        """
        Network with pseudo input
        """
        with tf.variable_scope('ensemble_model_pseudo'):
            logits_pseudo = self.model.build_logits(x_pseudo)
            log_liks_pseudo = self.model.build_log_liks_from_logits(
                logits_pseudo, y_pseudo)
            log_probs_pseudo = log_liks_pseudo+log_priors

        walk_loss = tf.reduce_mean(log_probs_pseudo)/self.config.train_data_size
        tf.add_to_collection('losses', walk_loss)

        """
        Walk process
        """
        def log_probs_pseudo_fn(_weights):
            return build_log_probs(
                _weights, x_pseudo, y_pseudo, self.model.layer_dims, self.model.data_size)

        walk_op, acpt_rate = mala(
            weights,
            log_probs_pseudo_fn,
            self.config.walk_size,
            self.config.num_walks_per_step)

        tf.add_to_collection('step_ops', walk_op)
        tf.add_to_collection('acpt_rate', acpt_rate)
        """
        Fisher divergence optimization
        """
        log_liks_grads = tf.gradients(log_liks, [weights])[0]
        log_liks_pseudo_grads = tf.gradients(log_liks_pseudo, [weights])[0]
        fisher = (log_liks_grads-log_liks_pseudo_grads)/self.config.train_data_size
        fisher = tf.reduce_mean(tf.square(fisher))
        fisher_opt_op = tf.train.GradientDescentOptimizer(
            learning_rate=self.config.fisher_learning_rate).minimize(
                fisher, var_list=[x_pseudo])

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

def mala(weights, log_probs_fn, lr, nb_walk_per_step):
    avg_acceptance_rate_ = 0.0
    weights_ = weights
    log_probs_ = log_probs_fn(weights_)
    grads_ = tf.gradients(log_probs_, weights_)[0]
    for step in range(nb_walk_per_step):
        eps = tf.random_normal(weights_.get_shape())
        proposed_weights_ = tf.stop_gradient(weights_+lr*grads_+tf.sqrt(2*lr)*eps)
        proposed_log_probs_ = log_probs_fn(proposed_weights_)
        proposed_grads_ = tf.gradients(proposed_log_probs_, proposed_weights_)[0]

        # add rejection step
        log_numer = proposed_log_probs_-0.25/lr*tf.reduce_sum(
            tf.square(weights_-proposed_weights_-lr*proposed_grads_), axis=1)
        log_denom = log_probs_-0.5*tf.reduce_sum(tf.square(eps), axis=1)
        acceptance_rate = tf.clip_by_value(tf.exp(log_numer-log_denom), 0.0, 1.0)

        # accept samples and update related quantities
        u = tf.random_uniform(acceptance_rate.get_shape())
        accept = tf.less_equal(u, acceptance_rate)
        weights_ = tf.where(accept, proposed_weights_, weights_)
        log_probs_ = tf.where(accept, proposed_log_probs_, log_probs_)

        avg_acceptance_rate_ += tf.reduce_mean(acceptance_rate)/nb_walk_per_step
        if step < nb_walk_per_step-1:
            grads_ = tf.where(accept, proposed_grads_, grads_)

    return tf.assign(weights, weights_), avg_acceptance_rate_

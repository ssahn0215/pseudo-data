import tensorflow as tf
import numpy as np

from models.base_model import BaseModel


def create_weights(layer_dims, ensemble_dim):
    weight_dim = 0
    for in_dim, out_dim in zip(layer_dims[:-1], layer_dims[1:]):
        weight_dim += (in_dim+1)*out_dim

    weights = tf.get_variable(
        name="weights",
        shape=[ensemble_dim, weight_dim],
        initializer=tf.initializers.random_normal)

    return weights

def build_logits(weights, x, layer_dims):
    ensemble_dim, weight_dim = weights.get_shape().as_list()
    _dim = 0

    act = tf.tile(tf.expand_dims(x, 0), [ensemble_dim, 1, 1])
    for l, (in_dim, out_dim) in enumerate(zip(
        layer_dims[:-1], layer_dims[1:])):
        with tf.variable_scope('dense'+str(l)):
            # collect corresponding weights
            kernel = tf.reshape(
                weights[:, _dim:_dim+in_dim*out_dim],
                [ensemble_dim, in_dim, out_dim])
            _dim += in_dim*out_dim
            bias = tf.expand_dims(weights[:, _dim:_dim+out_dim], 1)
            _dim += out_dim

            # do forward propagation
            pre = tf.matmul(act, kernel)+bias
            if l < len(layer_dims)-2:
                act = tf.nn.relu(pre)

    return pre

def build_log_liks_from_logits(weights, logits, y, data_size):
    ensemble_dim = weights.get_shape().as_list()[0]
    with tf.variable_scope('tile'):
        tiled_y = tf.tile(tf.expand_dims(y, 0), [ensemble_dim, 1, 1])

    with tf.variable_scope('log_liks'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=logits, labels=tiled_y)
        log_liks = -data_size*tf.reduce_mean(cross_entropy, axis=1)

    return log_liks

def build_log_priors(weights):
    with tf.variable_scope('log_priors'):
        log_priors = -0.5*tf.reduce_sum(tf.square(weights), 1)
    return log_priors

def build_log_liks(weights, x, y, layer_dims, data_size):
    logits = build_logits(
        weights, x, layer_dims)
    log_liks = build_log_liks_from_logits(
        weights, logits, y, data_size)
    return log_liks

def build_log_probs(weights, x, y, layer_dims, data_size):
    log_liks = build_log_liks(
        weights, x, y, layer_dims, data_size)
    log_priors = build_log_priors(weights)
    return log_liks+log_priors

class MnistEnsembleModel(BaseModel):
    def __init__(self, config):
        super(MnistEnsembleModel, self).__init__(config)

        # define important parameters
        self.ensemble_dim = config.ensemble_dim
        self.layer_dims = config.layer_dims
        self.data_size = config.train_data_size

        # pre-define important tf nodes
        self.weights = None
        self.logits = None
        self.log_liks = None
        self.log_priors = None
        self.log_probs = None

    def build_model(self, x, y, name='ensemble_model'):
        with tf.variable_scope(name):
            self.weights = create_weights(self.layer_dims, self.ensemble_dim)
            self.logits = self.build_logits(x)
            self.log_liks = build_log_liks_from_logits(
                self.weights, self.logits, y, self.config.train_data_size)
            self.log_priors = self.build_log_priors()
            self.log_probs = self.log_liks+self.log_priors

        return self.weights, self.logits, self.log_liks, self.log_priors, self.log_probs

    def build_logits(self, x):
        return build_logits(self.weights, x, self.layer_dims)

    def build_log_liks_from_logits(self, logits, y):
        return build_log_liks_from_logits(
            self.weights, logits, y, self.data_size)

    def build_log_liks(self, x, y):
        return build_log_liks(self.weights, x, y, self.layer_dims)

    def build_log_priors(self):
        return build_log_priors(self.weights)

    def build_log_probs(self, x, y):
        return build_log_probs(self.weights, x, y, self.layer_dims)

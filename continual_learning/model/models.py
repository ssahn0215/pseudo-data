import tensorflow as tf
import numpy as np
from copy import deepcopy
import sys
sys.path.extend(['alg/'])
import utils

np.random.seed(0)
tf.set_random_seed(0)

class Classification_NN(object):
    def __init__(self, model_size, batch_size):
        _ = model_size, batch_size, len(model_size)-1
        self.size, self.batch_size, self.nb_layers = _

        self.init_batch()
        self.init_params()
        self.init_layers()
        self.init_ops()

    def init_batch(self):
        self.X_ph = tf.placeholder(tf.float32, [None, self.size[0]])
        self.y_ph = tf.placeholder(tf.float32, [None, self.size[-1]])
        dataset = tf.data.Dataset.from_tensor_slices(
            (self.X_ph, self.y_ph)).batch(self.batch_size).repeat()
        self.iter = dataset.make_initializable_iterator()
        self.batch_X, self.batch_y = self.iter.get_next()

    def init_params(self):
        self.params = []

    def init_layers(self):
        pass

    def init_ops(self):
        self.param_phs = [
            tf.placeholder(param.dtype, shape=param.get_shape())
            for param in self.params]
        self.assign_param_ops = [
            tf.assign(p, ph)
            for p, ph in zip(self.params, self.param_phs)]

    def assign_data(self, sess, X, y):
        sess.run(
            self.iter.initializer,
            feed_dict={self.X_ph: X, self.y_ph: y})

    def assign_params(self, sess, params):
        sess.run(
            self.assign_param_ops,
            feed_dict={ph: p for ph, p in zip(self.param_phs, params)})

    def get_params(self, sess):
        return sess.run(self.params)

class Deterministic_NN(Classification_NN):
    def __init__(self, model_size, batch_size):
        super(Deterministic_NN, self).__init__(model_size, batch_size)

    def init_params(self):
        self.params = []
        for din, dout in zip(self.size[:-1], self.size[1:]):
            W_ = utils.mean_variable([din, dout])
            b_ = utils.mean_variable([1, dout])
            self.params.extend([W_, b_])

    def init_layers(self):
        pred = self.prediction_layer(self.batch_X)

        indiv_log_lik = tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=pred, labels=self.batch_y)
        self.loss = tf.reduce_mean(indiv_log_lik)

        self.indiv_log_pred = tf.nn.log_softmax(pred)
        pred_labels = tf.argmax(self.indiv_log_pred, axis=1)
        labels = tf.argmax(self.batch_y, axis=1)
        self.acc = tf.reduce_mean(
            tf.cast(tf.equal(labels, pred_labels), tf.float32))

    def prediction_layer(self, X):
        act = X
        #act = self.batch_X
        for i in range(self.nb_layers):
            W_, b_ = self.params[2*i:2*(i+1)]
            pre = tf.add(tf.matmul(act, W_), b_)
            if i < self.nb_layers-1:
                act = tf.nn.relu(pre)
        return pre

class Ensemble_NN(Classification_NN):
    def __init__(self, model_size, batch_size, nb_ensembles):
        self.nb_ensembles = nb_ensembles
        super(Ensemble_NN, self).__init__(model_size, batch_size)

    def init_params(self):
        self.params = []
        for din, dout in zip(self.size[:-1], self.size[1:]):
            W_ = utils.mean_variable([self.nb_ensembles, din, dout])
            b_ = utils.mean_variable([self.nb_ensembles, 1, dout])
            self.params.extend([W_, b_])

    def init_layers(self):
        ensembled_pred = self.prediction_layer(self.batch_X)
        indiv_log_lik = tf.reduce_sum(
            tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=ensembled_pred,
            labels=tf.tile(
                tf.expand_dims(self.batch_y, 0),
                [self.nb_ensembles, 1, 1])),
            axis = 0)
        self.loss = tf.reduce_mean(indiv_log_lik)

        self.indiv_log_pred = tf.reduce_mean(tf.nn.log_softmax(
            ensembled_pred), axis=0)
        pred_labels = tf.argmax(self.indiv_log_pred, axis=1)
        labels = tf.argmax(self.batch_y, axis=1)
        self.acc = tf.reduce_mean(
            tf.cast(tf.equal(labels, pred_labels), tf.float32))

    def prediction_layer(self, X):
        act = tf.tile(
            tf.expand_dims(X, 0),
            [self.nb_ensembles, 1, 1])
        for i in range(self.nb_layers):
            W_, b_ = self.params[2*i:2*(i+1)]
            pre = tf.add(tf.einsum('mni,mio->mno', act, W_), b_)
            if i < self.nb_layers - 1:
                act = tf.nn.relu(pre)
        return pre

class Stochastic_NN(Classification_NN):
    def __init__(self, model_size, batch_size, nb_ensembles):
        self.nb_ensembles = nb_ensembles
        super(Stochastic_NN, self).__init__(model_size, batch_size)

    def init_params(self):
        self.params = []
        for din, dout in zip(self.size[:-1], self.size[1:]):
            W_m_ = utils.mean_variable([din, dout])
            W_log_v_ = utils.small_variable([din, dout])
            b_m_ = utils.mean_variable([1, dout])
            b_log_v_ = utils.small_variable([1, dout])

            self.params.extend([W_m_, W_log_v_, b_m_, b_log_v_])

    def prediction_layer(self, nb_ensembles, X):
        act = tf.tile(tf.expand_dims(X, 0), [nb_ensembles, 1, 1])
        for i, (din, dout) in enumerate(zip(self.size[:-1], self.size[1:])):
            W_m_, W_log_v_, b_m_, b_log_v_ = self.params[4*i:4*(i+1)]

            W_eps_ = tf.random_normal([nb_ensembles, din, dout])
            W_ = tf.add(tf.multiply(tf.exp(0.5*W_log_v_), W_eps_), W_m_)

            b_eps_ = tf.random_normal([nb_ensembles, 1, dout])
            b_ = tf.add(tf.multiply(tf.exp(0.5*b_log_v_), b_eps_), b_m_)

            pre = tf.add(tf.einsum('mni,mio->mno', act, W_), b_)
            if i < self.nb_layers-1:
                act = tf.nn.relu(pre)

        return pre

    def init_layers(self):
        ensembled_pred = self.prediction_layer(self.nb_ensembles, self.batch_X)
        indiv_log_lik = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=ensembled_pred,
            labels=tf.tile(
                tf.expand_dims(self.batch_y, 0),
                [self.nb_ensembles, 1, 1])),
            axis = 0)
        self.loss = tf.reduce_mean(indiv_log_lik)

        self.indiv_log_pred = tf.reduce_mean(
            tf.nn.log_softmax(ensembled_pred), axis=0)
        pred_labels = tf.argmax(self.indiv_log_pred, axis=1)
        labels = tf.argmax(self.batch_y, axis=1)
        self.acc = tf.reduce_mean(
            tf.cast(tf.equal(labels, pred_labels), tf.float32))

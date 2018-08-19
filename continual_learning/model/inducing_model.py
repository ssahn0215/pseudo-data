import sys
import time
import tensorflow as tf
import numpy as np
import utils
sys.path.extend(['optimizer/'])
import sgld

class Deterministic_NN(object):
    def __init__(self, config):
        self.size = size = config.size
        self.nb_layers = len(size)-1

    def create_w_vars(self):
        w_vars = []
        for din, dout in zip(self.size[:-1], self.size[1:]):
            W_ = utils.mean_variable([din, dout])
            b_ = utils.mean_variable([1, dout])
            w_vars.extend([W_, b_])

        return w_vars

    def prediction_layer(self, X):
        act = X
        for i in range(self.nb_layers):
            W_, b_ = self.w_vars[2*i:2*(i+1)]
            pre = tf.add(tf.matmul(act, W_), b_)
            if i < self.nb_layers-1:
                act = tf.nn.relu(pre)

        return pre

    def log_lik_layer(self, pred, y):
        indiv_log_lik = tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=pred, labels=y)
        log_lik = tf.reduce_mean(indiv_log_lik)

        return log_lik

    def accuracy_layer(self, pred, y):
        indiv_log_pred = tf.reduce_mean(tf.nn.log_softmax(pred), axis=0)
        pred_labels = tf.argmax(indiv_log_pred, axis=1)
        labels = tf.argmax(y, axis=1)
        acc = tf.reduce_mean(tf.cast(
            tf.equal(labels, pred_labels), tf.float32))

    def reg_loss_layer(self):
        loss = 0.0
        for w_var in self.w_vars:
            loss += 0.5*tf.reduce_sum(tf.square(w_var))

        return loss/self.data_size

class Ensemble_NN(Deterministic_NN):
    def __init__(self, config):
        super(Ensemble_NN, self).__init__(config)

    def create_w_vars(self):
        w_vars = []
        for din, dout in zip(self.size[:-1], self.size[1:]):
            W_ = utils.mean_variable([self.nb_ensembles, din, dout])
            b_ = utils.mean_variable([self.nb_ensembles, 1, dout])
            w_vars.extend([W_, b_])

        return w_vars

    def prediction_layer(self, X):
        act = tf.tile(
            tf.expand_dims(X, 0),
            [self.nb_ensembles, 1, 1])
        for i in range(self.nb_layers):
            W_, b_ = self.w_vars[2*i:2*(i+1)]
            pre = tf.add(tf.einsum('mni,mio->mno', act, W_), b_)
            if i < self.nb_layers - 1:
                act = tf.nn.relu(pre)
        return pre

    def log_lik_layer(self, pred, y):
        indiv_log_lik = tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=pred,
            labels=tf.tile(
                tf.expand_dims(y, 0),
                [self.nb_ensembles, 1, 1]))
        log_lik = tf.reduce_mean(tf.reduce_sum(indiv_log_lik, axis=0), axis=0)
        return log_lik

class Training_Model(Deterministic_NN):
    def __init__(self, config):
        super(Training_Model, self).__init__(config)
        self.data_size = data_size = config.data_size
        self.batch_size = batch_size = config.batch_size
        self.nb_batchs = int(np.ceil(data_size/batch_size))
        self.init_lr = config.init_lr

    def build_graph(self):
        # Objects for data
        self.X_ph = tf.placeholder(tf.float32, [None, self.size[0]])
        self.y_ph = tf.placeholder(tf.float32, [None, self.size[-1]])
        dataset = tf.data.Dataset.from_tensor_slices(
            (self.X_ph, self.y_ph)).batch(self.batch_size).repeat()
        iter = dataset.make_initializable_iterator()
        self.batch_X, self.batch_y = iter.get_next()

        # Ops for data
        self.iter_init_op = iter.initializer

        # Variables to be trained
        self.w_vars = self.create_w_vars()

        # Main graph
        pred = self.prediction_layer(self.batch_X)
        log_lik = self.log_lik_layer(pred, self.batch_y)
        acc = self.accuracy_layer(pred, self.batch_y)

        # Gradients
        log_lik_grads = tf.gradients(log_lik, self.w_vars)
        reg_loss, reg_grads = self.reg_loss_and_grads()
        self.losses = [log_lik+reg_loss]

        grads = [g0+g1 for g0, g1 in zip(log_lik_grads, reg_grads)]

        lr = tf.Variable(self.init_lr)
        opt = tf.train.GradientDescentOptimizer(learning_rate=lr)
        self.train_op = opt.apply_gradients(
            list(zip(grads, self.w_vars)))

    def reg_loss_and_grads(self):
        loss = 0.0
        for w_var in self.w_vars:
            loss += 0.5*tf.reduce_sum(tf.square(w_var))

        loss /= self.data_size
        grads = tf.gradients(loss, self.w_vars)

        return loss, grads

class Inducing_Model(Ensemble_NN):
    def __init__(self, init_id_X, init_id_y, config):
        super(Inducing_Model, self).__init__(config)
        self.init_id_X, self.init_id_y = init_id_X, init_id_y
        self.data_size = data_size = config.data_size
        self.batch_size = batch_size = config.batch_size
        self.nb_batchs = int(np.ceil(data_size/batch_size))
        self.init_lr = config.init_lr

    def build_graph(self):
        # Objects for data
        self.X_ph = tf.placeholder(tf.float32, [None, self.size[0]])
        self.y_ph = tf.placeholder(tf.float32, [None, self.size[-1]])
        dataset = tf.data.Dataset.from_tensor_slices(
            (self.X_ph, self.y_ph)).batch(self.batch_size).repeat()
        iter = dataset.make_initializable_iterator()
        batch_X, batch_y = iter.get_next()

        # Ops for data
        self.iter_init_op = iter.initializer

        # Variables to be trained
        self.w_vars = self.create_w_vars()
        self.id_X = tf.Variable(self.init_id_X)
        self.id_y = tf.Variable(self.init_id_y)

        # Main graph
        pred = self.prediction_layer(self.batch_X)
        log_lik = self.log_lik_layer(pred, self.batch_y)
        acc = self.accuracy_layer(pred, self.batch_y)

        id_pred = self.prediction_layer(self.id_X)
        id_log_lik = self.log_lik(id_pred, self.id_y)

        # Gradients
        w_log_lik_grads = tf.gradients(log_lik, self.w_vars)
        w_id_log_lik_grads = tf.gradients(id_log_lik, self.w_vars)
        reg_loss, w_reg_grads = self.reg_loss_and_grads()
        loss = log_lik+reg_loss
        score_loss, id_score_grads = self.score_grads(
            w_log_lik_grads, id_log_lik_grads)

        self.losses = [loss, score_loss]


        w_grads = [g0+g1 for g0, g1 in zip(w_log_lik_grads, w_reg_grads)]
        w_id_grads = [g0+g1 for g0, g1 in zip(w_id_log_lik_grads, w_reg_grads)]

        # Build optimizers
        lr = tf.Variable(self.init_lr)
        w_opt = tf.train.GradientDescentOptimizer(
            learning_rate=self.lr)
        id_opt = tf.train.AdamOptimizer(learning_rate=self.lr)

        # Ops for optimization
        self.train_w_op = w_opt.apply_gradients(
            list(zip(w_grads, self.w_vars)))
        self.train_w_by_id_op = w_opt.apply_gradients(
            list(zip(w_id_grads, self.w_vars)))
        self.train_id_op = id_opt.apply_gradients(
            list(zip(id_score_grads, self.id_vars)))

    def reg_loss_and_grads(self):
        loss = 0.0
        for w_var in self.w_vars:
            loss += 0.5*tf.reduce_sum(tf.square(w_var))

        loss /= self.data_size
        grads = tf.gradients(loss, self.w_vars)

        return loss, grads

    def score_loss_and_grads(self, grads, id_grads):
        loss = tf.reduce_sum(
            [tf.reduce_sum(tf.square(id_grad-grad))
             for grad, id_grad in zip(grads, id_grads)])

        grads = tf.gradients(loss, self.id_X)

        return loss, grads

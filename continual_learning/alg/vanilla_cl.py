import tensorflow as tf
import numpy as np
import time
import sys
import utils

class Vanilla_CL():
    def __init__(
        self,
        model,
        reg_model,
        optimizer,
        lam,
        lr_fun,
        data_size):
        _ = model, reg_model, optimizer, lam, lr_fun, data_size
        self.model, self.reg_model, self.optimizer, self.lam, self.lr_fun, self.data_size = _
        self.init_params_and_ops()

    ### Initializing functions
    def init_params_and_ops(self):
        self.prev_model_params, self.prev_reg_model_params = None, None

        ## Define variables
        self.global_step = tf.Variable(0, trainable=False)
        self.lr = self.lr_fun(self.global_step)
        self.opt = self.optimizer(self.lr)

        #self.increment_global_step_op = tf.assign_add(self.global_step, 1)

    ### Pre task functions
    def pre_task_updates(self):
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        if self.prev_model_params:
            self.model.assign_params(self.sess, self.prev_model_params)
        if self.prev_reg_model_params:
            self.reg_model.assign_params(self.sess, self.prev_reg_model_params)

        self.update_loss_and_ops()

    def update_loss_and_ops(self):
        self.loss = (
            self.model.loss
            +self.lam*self.reg_model.deploy_reg_loss(self.sess)/self.data_size)

        grads_and_vars = self.opt.compute_gradients(
            self.loss, var_list=self.model.params)
        self.model_step_op = self.opt.apply_gradients(
            grads_and_vars, global_step=self.global_step)

        self.step_ops = [self.model_step_op]
        self.stat_vars = [self.global_step, self.model.acc, self.loss]

        self.sess.run(tf.variables_initializer(self.opt.variables()))

    ### Post task functions
    def post_task_updates(self, X, y, next_task=False):
        if next_task:
            self.fit_reg()
            self.save_params()

        self.sess.close()

    def fit_reg(self):
        pass

    def save_params(self):
        self.prev_model_params = self.model.get_params(self.sess)
        self.prev_reg_model_params = self.reg_model.get_params(self.sess)

    ### Main functions
    def fit(self, X, y, nb_epoch):
        data_size = X.shape[0]
        nb_batch = int(np.ceil(data_size/self.model.batch_size))
        tic = time.time()
        for epoch in range(1, nb_epoch+1):
            perm_ind = np.random.permutation(data_size)
            self.model.assign_data(self.sess, X[perm_ind, :], y[perm_ind, :])

            avg_loss, avg_acc = 0.0, 0.0
            for i in range(nb_batch):
                _, stats = self.sess.run(
                    [self.step_ops, self.stat_vars])

                global_step, acc, loss = stats
                avg_acc += stats[1]/nb_batch
                avg_loss += stats[2]/nb_batch

            if epoch%np.max([1, int(nb_epoch/10)]) == 0:
                toc = time.time()
                print("Epoch {:2d}, time={:.2f}, "
                      .format(epoch, toc-tic), end='')
                print("global step={:2d}, acc={:.3f}, loss={:.6f}"
                      .format(global_step, avg_acc, avg_loss))
                tic = toc

    def evaluate(self, X, y):
        tic = time.time()
        acc = utils.evaluate(self.sess, self.model, X, y)
        toc = time.time()
        print("Acc={:.3f}, time={:.2f}".format(acc, toc-tic))

        return acc

import tensorflow as tf
import numpy as np
import time
from vanilla_cl import Vanilla_CL
import sys
sys.path.extend(['model/'])
import reg_models

class Fisher_CL(Vanilla_CL):
    def __init__(
        self,
        model,
        reg_model,
        optimizer,
        lam,
        lr_fun,
        data_size,
        fisher_batch_size=60,
        nb_fisher_batch=1000):
        _ = fisher_batch_size, nb_fisher_batch
        self.fisher_batch_size, self.nb_fisher_batch = _
        super(Fisher_CL, self).__init__(
            model, reg_model, optimizer, lam, lr_fun, data_size)

    def init_params_and_ops(self):
        self.init_fisher_ops()
        super(Fisher_CL, self).init_params_and_ops()

    def init_fisher_ops(self):
        self.X_fisher_ph = tf.placeholder(
            tf.float32, [self.fisher_batch_size, self.model.size[0]])
        self.y_fisher_ph = tf.placeholder(
            tf.float32, [self.fisher_batch_size, self.model.size[-1]])
        fisher_indiv_log_lik = tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=self.model.prediction_layer(self.X_fisher_ph),
            labels=self.y_fisher_ph)

        fisher_log_liks = tf.unstack(fisher_indiv_log_lik)
        square_grads = [tf.zeros_like(param) for param in self.model.params]
        for fisher_log_lik in fisher_log_liks:
            grads = tf.gradients(fisher_log_lik, self.model.params)
            for i, grad in enumerate(grads):
                square_grads[i] += grad**2

        self.fisher = [
            tf.Variable(tf.zeros_like(param)) for param in self.model.params]
        self.zero_fisher_ops = [
            tf.assign(f, tf.zeros_like(f)) for f in self.fisher]
        self.cum_fisher_ops = [
            tf.assign_add(f, sg) for f, sg in zip(self.fisher, square_grads)]

    def post_task_updates(self, X, y, next_task=False):
        if next_task:
            self.fit_reg(X, y)
            self.save_params()

        self.sess.close()

    def fit_reg(self, X, y):
        new_params, cur_reg_params = self.sess.run(
            [self.model.params, self.reg_model.params])
        fisher = self.compute_fisher(X,y)

        new_reg_params = []
        for new_m_, cur_log_v_, f in zip(
            new_params, cur_reg_params[1::2], fisher):
            new_log_v_ = -np.log(np.exp(-cur_log_v_)+f)
            new_reg_params.extend([new_m_, new_log_v_])

        self.reg_model.assign_params(self.sess, new_reg_params)

    def compute_fisher(self, X, y):
        self.sess.run(self.zero_fisher_ops)
        data_size = X.shape[0]
        for i in range(self.nb_fisher_batch):
            start_idx = i*self.fisher_batch_size
            end_idx = (i+1)*self.fisher_batch_size
            if end_idx > data_size:
                break
            self.sess.run(
                self.cum_fisher_ops,
                feed_dict={
                    self.X_fisher_ph: X[start_idx:end_idx,:],
                    self.y_fisher_ph: y[start_idx:end_idx,:]})
        fisher = self.sess.run(self.fisher)
        fisher = [f / self.fisher_batch_size*self.nb_fisher_batch
                  for f in fisher]
        return fisher

class Fisher_Score_Matching_CL(Fisher_CL):
    def __init__(
        self,
        model,
        reg_model,
        optimizer,
        lam,
        lr_fun,
        data_size,
        fisher_batch_size=60,
        nb_fisher_batch=1000):
        super(Fisher_Score_Matching_CL, self).__init__(
            model, reg_model, optimizer, lam,
            lr_fun, data_size, fisher_batch_size, nb_fisher_batch)

    def update_loss_and_ops(self):
        self.loss = (
            self.model.loss
            +self.lam*self.reg_model.deploy_reg_loss(self.sess)/self.data_size)

        grads_and_vars = self.opt.compute_gradients(
            self.loss, var_list=self.model.params)
        self.grads, _ = zip(*grads_and_vars)
        self.model_step_op = self.opt.apply_gradients(
            grads_and_vars, global_step=self.global_step)

        self.step_ops = [self.model_step_op]
        self.stat_vars = [self.global_step, self.model.acc, self.loss]

        self.sess.run(tf.variables_initializer(self.opt.variables()))

    def fit_reg(self, X, y):
        data_size = X.shape[0]
        nb_batch = int(np.ceil(data_size/self.model.batch_size))
        means, cur_reg_params = self.sess.run(
            [self.model.params, self.reg_model.params])
        #log_vs = [0.0 for param in self.model.params]
        log_vs = [-tf.log(tf.exp(-log_v_)+f) for log_v_, f in
                  zip(cur_reg_params[1::2], self.compute_fisher(X,y))]

        self.reg_model = reg_models.Fisher_Gaussian_Score_Matching_Regularizer(
            self.model.params, self.grads, means, log_vs, float(data_size))

        self.sess.run(tf.variables_initializer([
            *self.reg_model.denoms,
            *self.reg_model.numers,
            *self.reg_model.params]))

        nb_epochs = 10
        for i in range(nb_epochs):
            print(i)
            perm_ind = np.random.permutation(data_size)
            self.model.assign_data(self.sess, X[perm_ind, :], y[perm_ind, :])
            for i in range(nb_batch):
                self.sess.run(self.reg_model.step_op)

        self.sess.run(self.reg_model.update_op)





'''
class EWC_CL(Fisher_CL):
    def __init__(
        self,
        model,
        reg_model,
        optimizer,
        lam,
        lr_fun,
        data_size,
        fisher_batch_size=60,
        nb_fisher_batch=1000):
        super(EWC_Optimizer_Constructor, self).__init__(
            model,
            optimizer,
            fisher_batch_size,
            nb_fisher_batch)

    def fit_reg(self, X, y):
        new_ms = self.sess.run(self.model.params)
        new_log_vs = [-np.log(f) for f in self.compute_fisher(X,y)]
        new_reg_params = []
        for m_, log_v_ in zip(new_ms, new_log_vs):
            new_reg_params.extend([m_, log_v_])

        self.reg_models.params.append(new_reg_params)
'''

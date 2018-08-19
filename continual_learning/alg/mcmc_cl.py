import tensorflow as tf
import numpy as np
import time
import utils
from vanilla_cl import Vanilla_CL
from models import Stochastic_NN, Deterministic_NN

class Vanilla_MCMC_CL(Vanilla_CL):
    def __init__(
        self,
        model,
        reg_model,
        optimizer,
        lam,
        lr_fun,
        data_size,
        burnin,
        thinning):

        _ = burnin, thinning, []
        self.burnin, self.thinning, self.params_samples = _
        super(Vanilla_MCMC_CL, self).__init__(
            model, reg_model, optimizer, lam, lr_fun, data_size)

    def init_params_and_ops(self):
        self.prev_model_params, self.prev_reg_model_params = None, None

        ## Define variables
        self.global_step = tf.Variable(0, trainable=False)
        self.lr = self.lr_fun(self.global_step)

        ## Define ops
        self.opt = self.optimizer(
            self.lr,
            data_size=self.data_size,
            burnin=self.burnin)

    ### Pre task functions
    def pre_task_updates(self):
        self.samples = []
        super(Vanilla_MCMC_CL, self).pre_task_updates()


    def update_loss_and_ops(self, remove_model_loss=False):
        if remove_model_loss:
            self.loss = self.reg_model.deploy_reg_loss(self.sess)/self.data_size
        else:
            self.loss = (
                self.model.loss
                +self.lam*self.reg_model.deploy_reg_loss(self.sess)/self.data_size)

        grads_and_vars = self.opt.compute_gradients(
            self.loss, var_list=self.model.params)
        self.model_step_op = self.opt.apply_gradients(
            grads_and_vars, global_step=self.global_step)

        self.step_ops = [self.model_step_op]
        self.stat_vars = [self.global_step, self.model.acc, self.loss, self.model.loss]
        self.sample_vars = [self.lr, self.model.params]

        self.sess.run(tf.variables_initializer(
            [*self.opt.variables(), self.opt._counter]))

    def fit(self, X, y, nb_epoch):
        data_size = X.shape[0]
        nb_batch = int(np.ceil(data_size/self.model.batch_size))
        tic = time.time()
        for epoch in range(0, nb_epoch+1):
            perm_ind = np.random.permutation(data_size)
            self.model.assign_data(self.sess, X[perm_ind, :], y[perm_ind, :])

            avg_loss, avg_acc, avg_model_loss = 0.0, 0.0, 0.0
            for i in range(nb_batch):
                if epoch > 0:
                    _, stats, samples = self.sess.run(
                        [self.step_ops, self.stat_vars, self.sample_vars])
                else:
                    stats = self.sess.run(self.stat_vars)

                global_step, acc, loss, model_loss = stats
                avg_acc += acc/nb_batch
                avg_loss += loss/nb_batch
                avg_model_loss += model_loss/nb_batch
                if (global_step+1>self.burnin and global_step%self.thinning==0):
                    self.samples.append(samples)

            if epoch%np.max([1, int(nb_epoch/10)]) == 0:
                toc = time.time()
                print("Epoch {:2d}, time={:.2f}, "
                      .format(epoch, toc-tic), end='')
                print("global step={:2d}, acc={:.3f}, loss={:.3f}, reg_loss={:.3f}"
                      .format(
                          global_step, avg_acc, avg_loss,
                          (avg_loss-avg_model_loss)*self.data_size))
                tic = toc

    def evaluate(self, X, y):
        tic = time.time()
        lr_samples, param_samples, _ = zip(*self.samples)
        acc = utils.evaluate(
             self.sess, self.model, X, y,
            params_list=param_samples[::100],
            weights=lr_samples[::100])
        toc = time.time()
        print("Acc={:.3f}, time={:.2f}".format(acc, toc-tic))

        return acc

class Gaussian_MCMC_CL(Vanilla_MCMC_CL):
    def __init__(
        self,
        model,
        reg_model,
        optimizer,
        lam,
        lr_fun,
        data_size,
        burnin,
        thinning):
        super(Gaussian_MCMC_CL, self).__init__(
            model, reg_model, optimizer, lam, lr_fun, data_size, burnin, thinning)

    def update_loss_and_ops(self):
        self.loss = (
            self.model.loss
            +self.lam*self.reg_model.deploy_reg_loss(self.sess)/self.data_size)

        grads_and_vars = self.opt.compute_gradients(
            self.loss, var_list=self.model.params)
        self.model_step_op = self.opt.apply_gradients(
            grads_and_vars, global_step=self.global_step)
        grads, _ = zip(*grads_and_vars)

        self.step_ops = [self.model_step_op]
        self.stat_vars = [self.global_step, self.model.acc, self.loss, self.model.loss]
        self.sample_vars = [self.lr, self.model.params, grads]

        self.sess.run(tf.variables_initializer(
            [*self.opt.variables(), self.opt._counter]))

    '''
    def fit_reg(self):
        global_step = 0
        nb_epochs = 100
        nb_samples = len(self.samples)
        #print(len(zip(*self.samples)))
        for epoch in range(1, nb_epochs):
            avg_loss = 0.0
            for lr, params, grads in self.samples:
                w = lr*100
                phs = [
                    self.reg_model.lr_ph,
                    *self.reg_model.target_param_phs,
                    *self.reg_model.target_grad_phs]
                vals = [w, *params, *grads]
                _, loss = self.sess.run(
                    [self.reg_model.cum_grad_ops, self.reg_model.loss],
                    feed_dict={ph: val for ph, val in zip(phs, vals)})
                avg_loss += lr*loss/nb_samples

            print(avg_loss)

            self.sess.run([
                *self.reg_model.assign_grad_ops,
                self.reg_model.zero_grad_ops])
    '''

    def fit_reg(self):
        lr_samples, param_samples, _ = zip(*self.samples)
        lr_samples = np.array(lr_samples)[:, np.newaxis, np.newaxis]

        reg_params = []
        for i in range(len(self.model.params)):
            param_i_samples = np.stack([p[i] for p in param_samples])
            weighted_avg = (
                np.sum(lr_samples*param_i_samples, axis=0)
                /np.sum(lr_samples, axis=0))
            weighted_var = (
                np.sum(lr_samples*(param_i_samples**2), axis=0)
                /np.sum(lr_samples, axis=0))
            weighted_var -= weighted_avg**2
            reg_params.extend([weighted_avg, 1/weighted_var])

        self.reg_model.assign_params(self.sess, reg_params)

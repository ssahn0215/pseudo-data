import tensorflow as tf
import numpy as np
import time
from vanilla_cl import Vanilla_CL
import sys
sys.path.extend(['model/'])
from gradient_replay_model import GradientReplay

class Gradient_Replay_CL(Vanilla_CL):
    def __init__(
        self,
        model,
        reg_model,
        optimizer,
        lam,
        lr_fun,
        data_size):
        super(Gradient_Replay_CL, self).__init__(
            model, reg_model, optimizer, lam, lr_fun, data_size)
        self.replay_models = []

    def update_loss_and_ops(self):
        self.loss = self.model.loss

        grads_and_params = self.opt.compute_gradients(
            self.loss, var_list=self.model.params)
        self.grads, _ = zip(*grads_and_params)
        self.grads = list(self.grads)
        for replay_model in self.replay_models:
            for i, grad in enumerate(replay_model.pred_grads):
                self.grads[i] += self.lam*grad

        self.model_step_op = self.opt.apply_gradients(
            zip(self.grads, self.model.params), global_step=self.global_step)

        self.step_ops = [self.model_step_op]
        self.stat_vars = [self.global_step, self.model.acc, self.loss]

        self.sess.run(tf.variables_initializer(self.opt.variables()))

    def post_task_updates(self, X, y, next_task=False):
        if next_task:
            self.fit_reg(X, y)
            self.save_params()

        self.sess.close()

    def fit_reg(self, X, y):
        data_size = X.shape[0]
        nb_batch = int(np.ceil(data_size/self.model.batch_size))
        init_params = self.model.get_params(self.sess)
        new_replay_model = GradientReplay(
            self.model.params,
            self.grads,
            float(data_size),
            lr=600.0, target_lr=1e-3,
            ref_params=init_params)
        self.sess.run(tf.variables_initializer(
            [*new_replay_model.params, *new_replay_model.cum_param_grads]))
        replay_step_ops = [
            #*new_replay_model.apply_grad_to_target_ops]
            *new_replay_model.apply_noise_to_target_ops]

        nb_rep = 30
        nb_epoch = 1
        for rep_idx in range(nb_rep):
            for epoch in range(1, nb_epoch+1):
                perm_ind = np.random.permutation(data_size)
                self.model.assign_data(self.sess, X[perm_ind, :], y[perm_ind, :])
                avg_loss = 0.0
                avg_grad_norm = 0.0
                avg_pred_grad_norm = 0.0
                for i in range(nb_batch):
                    self.model.assign_params(self.sess, init_params)
                    self.sess.run(new_replay_model.apply_noise_to_target_ops)
                    pred_grads, grads, loss = self.sess.run(
                        [new_replay_model.pred_grads, self.grads, new_replay_model.loss])
                    self.sess.run(new_replay_model.cum_grad_ops)
                    avg_loss += loss/nb_batch/epoch
                    avg_grad_norm += np.sum([np.sum(g**2) for g in grads])/nb_batch/epoch
                    avg_pred_grad_norm += np.sum([np.sum(g**2) for g in pred_grads])/nb_batch/epoch

                    self.sess.run(new_replay_model.apply_grad_ops)
                    self.sess.run(new_replay_model.zero_grad_ops)

                print('{:2d}/{:2d}/{:.6f}/{:.6f}/{:.6f}'.format(
                    rep_idx, epoch, avg_loss, avg_pred_grad_norm, avg_grad_norm))

        self.replay_models=[new_replay_model]

import tensorflow as tf
import numpy as np
import sys
sys.path.extend(['alg/'])
import utils

class GradientReplay(object):
    def __init__(
        self,
        target_params,
        target_grads,
        data_size,
        lr, target_lr,
        ref_params):
        self.target_params = target_params
        self.target_grads = target_grads
        self.target_shapes = [p.get_shape() for p in target_params]
        self.data_size = data_size
        self.lr = lr
        print(lr)
        self.target_lr = target_lr
        self.ref_params = ref_params

        self.init_params()
        self.init_layers()
        self.init_ops()


    def init_params(self):
        self.params, self.cum_param_grads = [], []
        for shape in self.target_shapes:
            #W_, b_ = tf.Variable(1e-3*tf.ones(shape)), tf.Variable(tf.zeros(shape))
            #W_grad_, b_grad_ = tf.Variable(tf.zeros(shape)), tf.Variable(tf.zeros(shape))
            #self.params.extend([W_, b_])
            #self.cum_param_grads.extend([W_grad_, b_grad_])
            self.params.append(tf.Variable(1.0*tf.ones(shape)))
            self.cum_param_grads.append(tf.Variable(tf.zeros(shape)))

    def init_layers(self):
        self.pred_grads = self.pred_layer()
        self.loss = self.loss_layer()
        self.param_grads = tf.gradients(self.loss, self.params)

    def pred_layer(self):
        pred = []
        #for param, W_, b_ in zip(
        #    self.target_params, self.params[0::2], self.params[1::2]):
        #    pred.append(W_*(param-b_))

        for param, W_, b_ in zip(
            self.target_params, self.params, self.ref_params):
            pred.append(W_*(param-b_))

        return pred

    def loss_layer(self):
        loss = 0.0
        for target_grad, pred_grad in zip(self.target_grads, self.pred_grads):
            loss += tf.reduce_sum(tf.square(target_grad-pred_grad))

        return loss

    def init_ops(self):
        self.apply_grad_to_target_ops = [
            tf.assign_add(param, self.target_lr*grad)
            for param, grad in zip(self.target_params, self.pred_grads)]

        target_noises = [
            tf.random_normal(shape=shape, mean=0.0, stddev=1.0)
            for shape in self.target_shapes]
        self.apply_noise_to_target_ops = [
            tf.assign_add(param, self.target_lr*eps)
            #tf.assign_add(param, tf.sqrt(self.target_lr*2.0/self.data_size)*eps)
            for param, eps in zip(self.target_params, target_noises)]

        self.cum_grad_ops, self.apply_grad_ops, self.zero_grad_ops = [], [], []
        for param, grad, cum_grad in zip(
            self.params, self.param_grads, self.cum_param_grads):
            apply_op_ = tf.assign_add(param, -self.lr*cum_grad)
            cum_grad_op_ = tf.assign_add(cum_grad, grad)
            zero_op_ = tf.assign(cum_grad, tf.zeros_like(cum_grad))

            self.apply_grad_ops.append(apply_op_)
            self.zero_grad_ops.append(zero_op_)
            self.cum_grad_ops.append(cum_grad_op_)

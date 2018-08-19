import tensorflow as tf
import numpy as np
import sys
sys.path.extend(['alg/'])
import utils

np.random.seed(0)
tf.set_random_seed(0)

class Regularizer(object):
    def __init__(self, target_params):
        self.target_params = target_params
        self.target_shapes = [p.get_shape() for p in target_params]

        self.init_params()
        self.init_ops()
        self.init_layers()


    def init_params(self):
        self.params = []

    def init_ops(self):
        self.param_phs = [
            tf.placeholder(tf.float32, shape=param.get_shape())
            for param in self.params]
        self.assign_param_ops = [
            tf.assign(p, ph)
            for p, ph in zip(self.params, self.param_phs)]

    def init_layers(self):
        pass

    def reg_loss_layer(self, params):
        return 0.0

    def deploy_reg_loss(self, sess):
        deployed_params = sess.run(self.params)
        return self.reg_loss_layer(self.target_params, deployed_params)

    def assign_params(self, sess, params):
        sess.run(
            self.assign_param_ops,
            feed_dict={ph: p for ph, p in zip(self.param_phs, params)})

    def get_params(self, sess):
        return sess.run(self.params)

class Gaussian_Regularizer(Regularizer):
    def __init__(self, target_params):
        super(Gaussian_Regularizer, self).__init__(target_params)

    def init_params(self):
        self.params = []
        for shape in self.target_shapes:
            m_, log_v_ = tf.Variable(tf.zeros(shape)), tf.Variable(tf.zeros(shape))
            self.params.extend([m_, log_v_])

    def reg_loss_layer(self, target_params, params):
        loss = 0.0
        for target_p_, m_, log_v_ in zip(
            target_params, params[0::2], params[1::2]):
            loss -= utils.tf_gaussian_log_prob(target_p_, m_, log_v_)

        return loss

'''
class Multi_Gaussian_Regularizer(Regularizer):
    def __init__(self, target_params):
        super(Gaussian_Regularizer, self).__init__(target_params)

    def init_params(self):
        self.params = []

    def reg_loss_layer(self, target_params, params):
        loss = 0.0
        for params_ in params:
            for target_p_, m_, log_v_ in zip(
                target_params, params_[0::2], params_[1::2]):
                loss -= utils.tf_gaussian_log_prob(target_p_, m_, log_v_)

        return loss

    def assign_params(self, sess, params):
        sess.run(
            self.assign_param_ops,
            feed_dict={
                ph: p for ph, p in zip(
                    self.param_phs, params[:len(self.param_phs)])})
'''



class KL_Gaussian_Regularizer(Regularizer):
    def __init__(self, target_params):
        super(KL_Gaussian_Regularizer, self).__init__(target_params)

    def init_params(self):
        self.params = []
        for shape in self.target_shapes:
            p_ = tf.zeros(shape)
            self.params.append(p_)

    def reg_loss_layer(self, target_params, params):
        loss = 0.0
        for target_m_, target_log_v_, m_, log_v_, in zip(
            target_params[0::2], target_params[1::2],
            params[0::2], params[1::2]):
            loss += utils.tf_gaussian_kl(
                target_m_, target_log_v_, m_, log_v_)

        return loss


class Score_Matching_Regularizer(Regularizer):
    def __init__(self,
                 target_params,
                 eps=1e-3):
        self.eps = eps
        super(Score_Matching_Regularizer, self).__init__(target_params)

    def init_ops(self):
        super(Score_Matching_Regularizer, self).init_ops()
        self.lr_ph = tf.placeholder(tf.float32, [])
        self.target_param_phs = [
            tf.placeholder(tf.float32, shape=shape)
            for shape in self.target_shapes]
        self.target_grad_phs = [
            tf.placeholder(tf.float32, shape=shape)
            for shape in self.target_shapes]

    def init_layers(self):
        self.delta = [utils.zeros_like_variable(p) for p in self.params]

        loss = self.reg_loss_layer(self.target_param_phs, self.params)
        grads = tf.gradients(loss, self.target_param_phs)
        grads_diff = [
            g1-g2 for g1, g2 in zip(grads, self.target_grad_phs)]
        self.loss = tf.reduce_sum([tf.reduce_sum(gd**2) for gd in grads_diff])

        pos_loss = self.reg_loss_layer(
            self.target_param_phs,
            [p+self.eps*gd for p, gd in zip(self.target_param_phs, grads_diff)])
        pos_grads = tf.gradients(pos_loss, self.params)

        neg_loss = self.reg_loss_layer(
            self.target_param_phs,
            [p-self.eps*gd for p, gd in zip(self.target_param_phs, grads_diff)])
        neg_grads = tf.gradients(neg_loss, self.params)

        self.cum_grad_ops = [
            tf.assign_add(d, self.lr_ph*0.5*(pg-ng)/self.eps)
            for d, pg, ng in zip(self.delta, pos_grads, neg_grads)]

        self.assign_grad_ops = [
            tf.assign_add(p, -d) for p, d in zip(self.params, self.delta)]
        with tf.control_dependencies(self.assign_grad_ops):
            self.zero_grad_ops = [
                tf.assign(d, tf.zeros(d.get_shape())) for d in self.delta]

class Fisher_Gaussian_Score_Matching_Regularizer(Regularizer):
    def __init__(
        self,
        target_params,
        target_grads,
        means,
        log_vs,
        data_size):
        _ = target_grads, means, log_vs, data_size
        self.target_grads, self.means, self.log_vs, self.data_size = _
        super(Fisher_Gaussian_Score_Matching_Regularizer, self).__init__(
            target_params)

    def init_params(self):
        self.params, self.denoms, self.numers, self.noises = [], [], [], []
        for shape, log_v_ in zip(self.target_shapes, self.log_vs):
            self.params.append(tf.Variable(tf.zeros(shape)))
            self.denoms.append(tf.Variable(tf.zeros(shape)))
            self.numers.append(tf.Variable(tf.zeros(shape)))

            self.noises.append(
                tf.exp(0.5*log_v_)*tf.random_normal(
                    shape=shape, mean=0.0, stddev=1.0))

    def init_ops(self):
        super(Fisher_Gaussian_Score_Matching_Regularizer, self).init_ops()
        apply_noise_ops = []
        for param, noise in zip(
            self.target_params, self.noises):
            apply_noise_ops.append(tf.assign_add(param, noise))

        cum_denom_ops, cum_numer_ops = [], []
        with tf.control_dependencies(apply_noise_ops):
            for param, mean, grad, denom, numer in zip(
                self.target_params, self.means,
                self.target_grads, self.denoms, self.numers):
                cum_denom_ops.append(tf.assign_add(denom, (param-mean)*grad))
                cum_numer_ops.append(tf.assign_add(numer, tf.square(param-mean)))

        with tf.control_dependencies([*cum_denom_ops, *cum_numer_ops]):
            reset_param_ops = []
            for param, mean in zip(self.target_params, self.noises):
                reset_param_ops.append(tf.assign(param, mean))

        self.step_op = tf.group(
            *apply_noise_ops,
            *cum_denom_ops,
            *cum_numer_ops,
            *reset_param_ops)

        update_param_ops = []
        for param, denom, numer in zip(self.params, self.denoms, self.numers):
            update_param_ops.append(
                tf.assign(param, tf.log(numer/denom/self.data_size)))

        zero_denom_and_numer_ops = []
        with tf.control_dependencies(update_param_ops):
            for denom, numer, shape in zip(
                self.denoms, self.numers, self.target_shapes):
                zero_denom_and_numer_ops.extend([
                    tf.assign(denom, tf.zeros(shape)),
                    tf.assign(numer, tf.zeros(shape))])

        self.update_op = tf.group(*update_param_ops, *zero_denom_and_numer_ops)

    def reg_loss_layer(self, target_params, params):
        loss = 0.0
        for target_p_, m_, log_v_ in zip(
            target_params, self.means, params):
            loss -= utils.tf_gaussian_log_prob(target_p_, m_, log_v_)

        return loss

    def assign_params(self, sess, params):
        print(params)
        params = [np.nan_to_num(param) for param in params]
        sess.run(
            self.assign_param_ops,
            feed_dict={ph: p for ph, p in zip(self.param_phs, params)})

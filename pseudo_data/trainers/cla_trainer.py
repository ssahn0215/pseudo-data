import tensorflow as tf
import numpy as np

from tqdm import tqdm
from utils.metrics import AverageMeter, MovingAverageMeter

class ClassifierTrainer:
    def __init__(self, sess, data_loader, logger, config):
        self.sess = sess
        self.data_loader = data_loader
        self.logger = logger
        self.config = config

    def train(self, num_steps):
        # initialize all variables
        self.sess.run(tf.global_variables_initializer())

        # get train related nodes
        global_step_tensor, = tf.get_collection('global_step')
        global_step_inc_op, train_op = tf.get_collection('step_ops')

        # get loss and error_nodes
        loss_node, = tf.get_collection('losses')
        error_node, = tf.get_collection('error')
        acpt_rate_node, = tf.get_collection('acpt_rate')

        # initialize tqdm
        init_global_step = self.sess.run(global_step_tensor)
        tt = tqdm(range(num_steps), total=num_steps,
            initial=init_global_step, miniters=1,
            desc="loss: {:.3E}, error: {:.2f}".format(0, 0))

        loss_per_epoch = MovingAverageMeter(50)
        error_per_epoch = MovingAverageMeter(50)
        acpt_rate_per_epoch = MovingAverageMeter(50)
        for cur_step in tt:
            # shuffle dataset every epoch
            if cur_step % self.data_loader.num_iterations_train == 0:
                self.data_loader.initialize(self.sess, state='train')

            _, _, loss, error, acpt_rate = self.sess.run([
                global_step_inc_op, train_op, loss_node, error_node, acpt_rate_node])

            loss_per_epoch.update(loss)
            error_per_epoch.update(error)
            acpt_rate_per_epoch.update(acpt_rate)

            tt.set_description("loss: {:.3E}, error: {:.2f}, acpt_rate: {:.2f} ".format(
                loss_per_epoch.avg,
                error_per_epoch.avg,
                acpt_rate_per_epoch.avg))

        tt.close()

    def test(self, state='test'):
        # get loss and error_nodes
        loss_node, = tf.get_collection('losses')
        error_node, = tf.get_collection('error')

        # initialize dataset
        self.data_loader.initialize(self.sess, state=state)

        # initialize tqdm
        tt = tqdm(range(self.data_loader.num_iterations_test), total=self.data_loader.num_iterations_test,
                  desc="test")

        loss_per_epoch = AverageMeter()
        error_per_epoch = AverageMeter()

        # Iterate over batches
        for _ in tt:
            # One Train step on the current batch
            loss, acc = self.sess.run([loss_node, error_node])

            # update metrics returned from train_step func
            loss_per_epoch.update(loss)
            error_per_epoch.update(acc)

        print("Test loss: {:.3E}, error: {:.2f}".format(
            loss_per_epoch.val, error_per_epoch.val))

        tt.close()
        return loss_per_epoch.val, error_per_epoch.val

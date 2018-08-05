import tensorflow as tf
import numpy as np

from tqdm import tqdm
from utils.metrics import MovingAverageMeter

class InducerTrainer:
    def __init__(self, sess, data_loader, logger, config):
        self.sess = sess
        self.data_loader = data_loader
        self.logger = logger
        self.config = config

    def train(self, num_steps):
        # initialize all variables
        self.sess.run(tf.global_variables_initializer())

        # load previously saved check points (if any)
        saver = tf.train.Saver(
            max_to_keep=self.config.max_to_keep, save_relative_paths=True)

        latest_checkpoint = tf.train.latest_checkpoint(self.config.checkpoint_dir)
        if latest_checkpoint:
            print("Loading model checkpoint {} ...".format(latest_checkpoint))
            saver.restore(self.sess, latest_checkpoint)

        # get train related nodes
        global_step_tensor = tf.get_collection('global_step')[0]
        global_step_inc_op, walk_op, fisher_opt_op = tf.get_collection('step_ops')

        # get loss and error_nodes
        walk_loss_node, fisher_loss_node = tf.get_collection('losses')
        error_node = tf.get_collection('error')[0]
        acpt_rate_node = tf.get_collection('acpt_rate')[0]

        # initialize tqdm
        init_global_step = self.sess.run(global_step_tensor)
        tt = tqdm(
            range(init_global_step, init_global_step+num_steps),
            total=self.config.num_training_steps,
            initial=init_global_step, miniters=1,
            desc="walk_loss: {:.3E}, error: {:.2f}".format(0, 0))

        walk_loss_per_epoch = MovingAverageMeter(self.data_loader.num_iterations_train)
        fisher_loss_per_epoch = MovingAverageMeter(self.data_loader.num_iterations_train)
        error_per_epoch = MovingAverageMeter(self.data_loader.num_iterations_train)
        acpt_rate_per_epoch = MovingAverageMeter(self.data_loader.num_iterations_train)

        for cur_step in tt:
            # shuffle dataset every epoch
            if cur_step % self.data_loader.num_iterations_train == 0:
                self.data_loader.initialize(self.sess, state='train')

                tt.set_description(
                    "walk_loss: {:.3E}, fisher_loss: {:.3E}, error: {:.2f}, acpt_rate: {:.2f}".format(
                        walk_loss_per_epoch.avg,
                        fisher_loss_per_epoch.avg,
                        error_per_epoch.avg,
                        acpt_rate_per_epoch.avg))

            if init_global_step+cur_step < self.config.num_burnin_steps:
                _, _, walk_loss, error = self.sess.run([
                    global_step_inc_op, walk_op, walk_loss_node, error_node])

                walk_loss_per_epoch.update(walk_loss)
                error_per_epoch.update(error)

                summaries_dict = {'train/walk_loss': walk_loss,
                                  'train/acc': error}
            else:
                _ = self.sess.run([
                    global_step_inc_op, walk_op, fisher_opt_op,
                    walk_loss_node, fisher_loss_node,
                    error_node, acpt_rate_node])
                _, _, _, walk_loss, fisher_loss, error, acpt_rate = _

                walk_loss_per_epoch.update(walk_loss)
                fisher_loss_per_epoch.update(fisher_loss)
                error_per_epoch.update(error)
                acpt_rate_per_epoch.update(acpt_rate)

                summaries_dict = {'train/walk_loss': walk_loss,
                                  'train/fisher_loss': fisher_loss,
                                  'train/acc': error,
                                  'train/acpt_rate': acpt_rate}

            self.logger.summarize(cur_step, summaries_dict)

        saver.save(self.sess, self.config.checkpoint_dir, global_step_tensor)
        tt.close()

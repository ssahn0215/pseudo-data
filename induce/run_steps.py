import tensorflow as tf
import numpy as np
import time

from tqdm import tqdm
from utils.metrics import MovingAverageMeter

def run_steps(num_steps, sess, data_loader, model, logger, config):
    # initialize all variables
    sess.run(tf.global_variables_initializer())

    # load previously saved check points (if any)
    saver = tf.get_collection('saver')[0]
    latest_checkpoint = tf.train.latest_checkpoint(config.checkpoint_dir)
    if latest_checkpoint:
        print("Loading model checkpoint {} ...\n".format(latest_checkpoint))
        saver.restore(sess, latest_checkpoint)
        print("Model loaded")

    # get inputs
    x, y = tf.get_collection('inputs')
    x_pseudo, y_pseudo = tf.get_collection('inputs_pseudo')

    # get train related nodes
    global_step_tensor = tf.get_collection('global_step')[0]
    global_step_inc_op, walk_op, fisher_opt_op = tf.get_collection('step_ops')

    # get loss and error_nodes
    walk_loss_node, fisher_loss_node = tf.get_collection('losses')
    error_node = tf.get_collection('error')[0]
    acpt_rate_node = tf.get_collection('acpt_rate')[0]

    # initialize tqdm
    init_global_step = sess.run(global_step_tensor)
    tt = tqdm(range(num_steps), total=num_steps,
        initial=init_global_step, miniters=1,
        desc="walk_loss: {:.3E}, error: {:.2f}".format(0, 0))

    walk_loss_per_epoch = MovingAverageMeter(50)
    fisher_loss_per_epoch = MovingAverageMeter(50)
    error_per_epoch = MovingAverageMeter(50)
    acpt_rate_per_epoch = MovingAverageMeter(50)
    for cur_step in tt:
        # shuffle dataset every epoch
        if cur_step % data_loader.num_iterations_train == 0:
            data_loader.initialize(sess, state='train')
            if cur_step > 0:
                print("Saving model...")
                saver.save(sess, config.checkpoint_dir, global_step_tensor)
                print("Model saved")

        if init_global_step+cur_step < config.num_burnin_steps:
            _, _, walk_loss, error = sess.run([
                global_step_inc_op, walk_op, walk_loss_node, error_node])
            walk_loss_per_epoch.update(walk_loss)
            error_per_epoch.update(error)
            tt.set_description(
                "walk_loss: {:.3E}, error: {:.2f} ".format(walk_loss, error))
        else:
            _ = sess.run([
                global_step_inc_op, walk_op, fisher_opt_op,
                walk_loss_node, fisher_loss_node,
                error_node, acpt_rate_node])
            _, _, _, walk_loss, fisher_loss, error, acpt_rate = _
            walk_loss_per_epoch.update(walk_loss)
            fisher_loss_per_epoch.update(fisher_loss)
            error_per_epoch.update(error)
            acpt_rate_per_epoch.update(acpt_rate)
            tt.set_description(
                "walk_loss: {:.3E}, fisher_loss: {:.3E}, error: {:.2f}, acpt_rate: {:.2f}".format(
                    error, walk_loss, error, acpt_rate))

    tt.close()
    print("walk_loss: {:.3E}, fisher_loss: {:.3E}, error: {:.2f}, acpt_rate: {:.2f}".format(
        error, walk_loss, error, acpt_rate))

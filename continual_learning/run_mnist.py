import numpy as np
import tensorflow as tf
import os
import sys
import time
import argparse

import utils
import models

parser = argparse.ArgumentParser()
parser.add_argument(
    '-gpu','--gpu-name', nargs='+',
    default=['0','1','2','3'],
    help="name of gpu to use")

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]=", ".join(args.gpu_name)
tf.reset_default_graph()
tf.set_random_seed(1)
np.random.seed(1)

def run_epochs(
    sess, model, X, y,
    global_step=0, nb_epochs=0, nb_burn_epochs=0,
    train_ops=None, burnin_ops=None):
    tic = time.time()
    for epoch in range(nb_epochs):
        perm_ind = np.random.permutation(model.data_size)
        sess.run(
            model.iter_init_op,
            feed_dict = {
                model.X_ph: X[perm_ind, :],
                model.y_ph: y[perm_ind, :]})

        fetches = {"losses": model.losses, "acc": model.acc}

        if epoch < nb_burn_epochs:
            fetches.update(burnin_ops)
        else:
            fetches.update(train_ops)

        losses, acc = [0.0]*len(model.losses), 0.0
        for _ in range(model.nb_batchs):
            global_step += 1
            vals = sess.run(fetches)
            losses = [c+nc for c, nc in zip(losses, vals["losses"])]
            acc += vals["acc"]/model.nb_batchs

        if epoch%np.max([1, int(nb_epochs/100)]) == 0:
            toc = time.time()
            print("Epoch {:04d}, time={:.1f}, global step={:2d}, acc={:.3f}, losses="
                  .format(epoch, toc-tic, global_step, acc), end='')
            print("".join('{:.4f}'.format(loss) for lost in losses))
            tic = toc

    return global_step


class Model_Config(object):
    size = [784, 100, 10]
    data_size = 60000
    batch_size = 1000
    id_batch_size = 100
    nb_ensembles= 256
    init_lr = 1e-2

class Exp_Config(object):
    nb_epochs0 = 20
    nb_epochs1 = 0
    nb_epochs2 = 5000
    nb_burn_epochs = 10

model_config, exp_config = Model_Config(), Exp_Config()
# Construct dataset
training_dataset, validation_dataset = utils.construct_permute_mnist(nb_tasks=1)
training_data, valid_data = training_dataset[0], validation_dataset[0]
train_X, train_y = training_data
valid_X, valid_y = valid_data

with tf.Session() as sess:
    model = model.Training_Model(model_config)
    model.build_graph()
    sess.run(tf.global_variables_initializer())
    run_epochs(
        sess, model, train_X, train_y,
        nb_epochs=exp_config.nb_epochs0,
        train_ops={"train":model.train_op})
    utils.evaluate(sess, model, valid_X, valid_y)

id_idx = np.random.choice(cur_batch_size, id_batch_size, replace=False)
init_id_X, init_id_y = batch_X[id_idx, :], batch_y[id_idx, :]

tf.reset_default_graph()
with tf.Session() as sess:
    model = model.Inducing_Model(model_config, init_id_X, init_id_y)
    model.build_graph()
    sess.run(tf.global_variables_initializer())
    run_epochs(
        sess, model, train_X, train_y,
        nb_epochs=exp_config.nb_epochs1,
        nb_burn_epochs=exp_config.nb_burn_epochs,
        train_ops={
            "train_w":model.train_w_by_id_op,
            "train_id":model.train_id_op},
        burnin_ops={"train_w":model.train_w_by_id_op})
    id_X, id_y = sess.run([model.id_X, model.id_y])

tf.reset_default_graph()
with tf.Session() as sess:
    model = model.Training_Model(model_config)
    model.build_graph()
    sess.run(tf.global_variables_initializer())
    run_epochs(
        sess, model, id_X, id_y,
        nb_epochs=exp_config.nb_epochs2,
        train_ops={"train":model.train_op})
    utils.evaluate(sess, model, valid_X, valid_y)

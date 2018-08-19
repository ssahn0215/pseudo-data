import gzip
import _pickle as cPickle
import pickle
import numpy as np
from copy import deepcopy
import tensorflow as tf
import os.path
import time

def construct_permute_mnist(nb_tasks=2,  split='train', permute_all=False, subsample=1):
    """Create permuted MNIST tasks.

        Args:
                nb_tasks: Number of tasks
                split: whether to use train or testing data
                permute_all: When set true also the first task is permuted otherwise it's standard MNIST
                subsample: subsample by so much

        Returns:
            List of (X, y) tuples representing each dataset
    """
    # Load MNIST data and normalize
    f = gzip.open('data/mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = cPickle.load(
        f, encoding='iso-8859-1')
    f.close()

    nb_classes = 10
    X_train = np.vstack((train_set[0], valid_set[0]))
    X_test = test_set[0]
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    y_train = np.eye(nb_classes)[np.hstack((train_set[1], valid_set[1]))]
    y_test = np.eye(nb_classes)[test_set[1]]
    y_train = y_train.astype('float32')
    y_test = y_test.astype('float32')

    permutations = []
    # Generate random permutations
    for i in range(nb_tasks):
        idx = np.arange(X_train.shape[1],dtype=int)
        if permute_all or i>0:
            np.random.shuffle(idx)
        permutations.append(idx)

    both_datasets = []
    for (X, y) in ((X_train, y_train), (X_test, y_test)):
        datasets = []
        for perm in permutations:
            data = X[:,perm], y
            datasets.append(data)
        both_datasets.append(datasets)

    return both_datasets

def tf_gaussian_log_prob(val, mean, log_var):
    var = tf.exp(log_var)
    return tf.reduce_sum(-0.5*(val-mean)**2/var)

def tf_gaussian_kl(m, log_v, m0, log_v0):
    log_std_diff = 0.5*tf.reduce_sum(log_v0 - log_v)
    mu_diff_term = 0.5*tf.reduce_sum((tf.exp(log_v) + (m0 - m)**2) / tf.exp(log_v0))
    kl = log_std_diff+mu_diff_term
    return kl

def zeros_like_variable(var, trainable=True):
    return tf.Variable(tf.zeros_like(var), trainable=trainable)


def mean_variable(shape, init_params=None):
    if init_params is not None:
        initial = tf.constant(init_params, shape=shape)
    else:
        initial = tf.truncated_normal(shape, stddev=0.1)

    return tf.Variable(initial)

def small_variable(shape):
    initial = tf.constant(-6.0, shape=shape)
    return tf.Variable(initial)

def save_zipped_pickle(obj, filename, protocol=-1):
    with gzip.open(filename, 'wb') as f:
        pickle.dump(obj, f, protocol)

def load_zipped_pickle(filename):
    try:
        with gzip.open(filename, 'rb') as f:
            loaded_object = pickle.load(f)
            return loaded_object
    except IOError:
        print("Warning: IO Error returning empty dict.")
        return dict()

def evaluate(sess, model, X, y):
    data_size = X.shape[0]
    nb_batch = int(np.ceil(data_size/model.batch_size))
    fetchs = {"batch_X": model.batch_X, "acc": model.acc}

    sess.run(model.iter_init_op, feed_dict={model.X_ph:X, model.y_ph:y})
    tic = time.time()
    acc = 0.0
    for batch_idx in range(nb_batch):
        vals = sess.run({"batch_X": model.batch_X, "acc": model.acc})
        acc += vals["acc"]*vals["batch_X"].shape[0]/data_size

    toc = time.time()
    print("Acc={:.3f}, time={:.2f}".format(acc, toc-tic))

    return acc, toc-tic

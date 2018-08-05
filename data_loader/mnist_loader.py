import pickle

from tqdm import tqdm

import tensorflow as tf
import numpy as np

class MnistDataLoader:
    """
    This will contain the dataset api
    It will load the numpy files from the pkl file which is dumped by prepare_mnist.py script
    Please make sure that you have included all of the needed config
    Thanks..
    """

    def __init__(self, config):
        self.config = config

        with open(self.config.data_numpy_pkl, "rb") as f:
            self.data_pkl = pickle.load(f)

        self.x_train = self.data_pkl['x_train']
        self.y_train = self.data_pkl['y_train']
        self.x_validate = self.data_pkl['x_validate']
        self.y_validate = self.data_pkl['y_validate']
        self.x_test = self.data_pkl['x_test']
        self.y_test = self.data_pkl['y_test']

        self.train_len = self.x_train.shape[0]
        self.validate_len = self.x_validate.shape[0]
        self.test_len = self.x_test.shape[0]

        self.num_iterations_train = (self.train_len + self.config.batch_size - 1) // self.config.batch_size
        self.num_iterations_validate = (self.validate_len + self.config.batch_size - 1) // self.config.batch_size
        self.num_iterations_test = (self.test_len + self.config.batch_size - 1) // self.config.batch_size

        self.features_placeholder = None
        self.labels_placeholder = None

        self.dataset = None
        self.iterator = None
        self.init_iterator_op = None
        self.next_batch = None

        self.build_dataset_api()

    def build_dataset_api(self):
        with tf.device('/cpu:0'):
            self.features_placeholder = tf.placeholder(tf.float32, [None] + list(self.x_train.shape[1:]))
            self.labels_placeholder = tf.placeholder(tf.int64, [None, self.y_train.shape[1]])

            self.dataset = tf.data.Dataset.from_tensor_slices((self.features_placeholder, self.labels_placeholder))
            self.dataset = self.dataset.repeat().batch(self.config.batch_size)

            self.iterator = tf.data.Iterator.from_structure(self.dataset.output_types, self.dataset.output_shapes)

            self.init_iterator_op = self.iterator.make_initializer(self.dataset)

            self.next_batch = self.iterator.get_next()

    def initialize(self, sess, state):
        if state == 'train':
            idx = np.random.choice(self.train_len, self.train_len, replace=False)
            self.x_train = self.x_train[idx]
            self.y_train = self.y_train[idx]
            sess.run(self.init_iterator_op, feed_dict={self.features_placeholder: self.x_train,
                                                       self.labels_placeholder: self.y_train})
        elif state == 'validate':
            sess.run(self.init_iterator_op, feed_dict={self.features_placeholder: self.x_validate,
                                                       self.labels_placeholder: self.y_validate})

        else:
            sess.run(self.init_iterator_op, feed_dict={self.features_placeholder: self.x_test,
                                                       self.labels_placeholder: self.y_test})

    def get_input(self):
        return self.next_batch

    def get_pseudo_input(self):
        if self.config.pseudo_init_method == "sample":
            x_pseudo, y_pseudo = [], []
            for c in range(self.config.num_classes):
                c_idx = np.where(np.argmax(self.y_train, axis=1)==c)[0]
                x_pseudo.append(self.x_train[c_idx[:int(self.config.pseudo_data_size/self.config.num_classes)],:])
                y_pseudo.append(self.y_train[c_idx[:int(self.config.pseudo_data_size/self.config.num_classes)],:])

            x_pseudo = np.concatenate(x_pseudo, axis=0)
            y_pseudo = np.concatenate(y_pseudo, axis=0)
            return x_pseudo, y_pseudo

        elif self.config.pseudo_init_method == "noise_with_int_lables":
            x_pseudo, y_pseudo = [], []
            for c in range(self.config.num_classes):
                c_idx = np.where(np.argmax(self.y_train, axis=1)==c)[0]
                x_pseudo.append(np.random.normal(
                    size=[int(self.config.pseudo_data_size/self.config.num_classes), 784]).astype(np.float32))
                y_pseudo.append(self.y_train[c_idx[:int(self.config.pseudo_data_size/self.config.num_classes)],:])

            x_pseudo = np.concatenate(x_pseudo, axis=0)
            y_pseudo = np.concatenate(y_pseudo, axis=0)
            return x_pseudo, y_pseudo


def main():
    class Config:
        data_numpy_pkl = "../data/mnist/data_numpy.pkl"
        data_mode = "numpy"

        image_height = 28
        image_width = 28
        batch_size = 8

        pseudo_dim = 1000
        init_pseudo_x = "noise"
        init_pseudo_y = "rand_int"

    tf.reset_default_graph()

    sess = tf.Session()

    data_loader = MnistDataLoaderNumpy(Config)

    x, y = data_loader.next_batch

    data_loader.initialize(sess, state='train')

    out_x, out_y = sess.run([x, y])

    print(out_x.shape, out_x.dtype)
    print(out_y.shape, out_y.dtype)

    data_loader.initialize(sess, state='validate')

    out_x, out_y = sess.run([x, y])

    print(out_x.shape, out_x.dtype)
    print(out_y.shape, out_y.dtype)

    data_loader.initialize(sess, state='test')

    out_x, out_y = sess.run([x, y])

    print(out_x.shape, out_x.dtype)
    print(out_y.shape, out_y.dtype)

if __name__ == '__main__':
    main()

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
import os

def reshape_and_tile_images(array, shape=(28, 28), n_cols=None):
    if n_cols is None:
        n_cols = int(math.sqrt(array.shape[0]))
    n_rows = int(np.ceil(float(array.shape[0])/n_cols))
    def cell(i, j):
        ind = i*n_cols+j
        if i*n_cols+j < array.shape[0]:
            return np.pad(
                array[ind].reshape(*shape, order='C'),
                pad_width=3,
                mode='constant',
                constant_values=1.0)
        else:
            return np.zeros(shape)

    def row(i):
        return np.concatenate([cell(i, j) for j in range(n_cols)], axis=1)

    return np.concatenate([row(i) for i in range(n_rows)], axis=0)

def plot_images(images, filename, dirname, shape=(28, 28), n_rows=10):
    images = reshape_and_tile_images(images, shape, n_rows)
    plt.imsave(fname=os.path.join(dirname+filename), arr=images, cmap=cm.Greys_r)
    plt.close()

def plot_images_and_labels(images, labels, filename, dirname, shape=(28, 28), num_classes=10, n_rows=10):
    shape = (shape[0]+8, shape[1])
    labels /= np.max(labels, axis=1, keepdims=True)
    labels = np.repeat(labels, int(shape[0]/num_classes), axis=1)[:, :shape[1]]
    labels = np.tile(labels, 5)
    images = np.concatenate([
        images, np.ones([images.shape[0], 28*3]), labels], axis=1)
    images = reshape_and_tile_images(images, shape, n_rows)
    plt.imsave(fname=os.path.join(dirname+filename), arr=images, vmin=0.0, vmax=0.5, cmap=cm.Greys_r)
    plt.close()

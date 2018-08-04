import sys
import os

import tensorflow as tf
import numpy as np

from build import build
from run_steps import run_steps

sys.path.extend([os.path.join(sys.path[0],'..')])

from data_loader.mnist_loader import MnistDataLoader

from models.ensemble_model import MnistEnsembleModel

from utils.config import process_config
from utils.dirs import create_dirs
from utils.logger import DefinedSummarizer
from utils.utils import get_args

def induce(num_steps, sess, config):
    # make tensorflow session
    sess = tf.Session()
    # create the experiments dirs
    create_dirs([config.summary_dir, config.checkpoint_dir, config.etc_dir])

    # create tensorboard logger
    logger = DefinedSummarizer(
        sess, summary_dir=config.summary_dir,
        scalar_tags=['train/loss_per_epoch', 'train/acc_per_epoch'])

    # get data loader
    data_loader = MnistDataLoader(config)

    # get model
    model = MnistEnsembleModel(config)

    # build graph
    build(data_loader, model, config)

    # train inducing points
    run_steps(num_steps, sess, data_loader, model, logger, config)

    # get outputs
    sess.run(tf.get_collection('id_inputs'))

    # close session
    sess.close()

    return

def main():
    # capture the config path from the run arguments
    # then process the json configration file
    #try:
    args = get_args()
    #print(args)
    config = process_config(args.config)

    #except:
    #    print("missing or invalid arguments")
    #    exit(0)

    # set visible device
    os.environ["CUDA_VISIBLE_DEVICES"]=", ".join(args.gpu_name)

    # initialize tensorflow graph
    tf.reset_default_graph()

    # fix random seeds
    tf.set_random_seed(1)
    np.random.seed(1)

    # run induce function
    sess = tf.Session()
    induce(config.num_training_steps, sess, config)
    sess.close()

if __name__ == '__main__':
    main()

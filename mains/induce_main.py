import sys
import os

import tensorflow as tf
import numpy as np

import pickle

sys.path.extend([os.path.join(sys.path[0],'..')])

from data_loader.mnist_loader import MnistDataLoader

from models.ensemble_model import MnistEnsembleModel

from builders.induce_builder import InducerBuilder

from trainers.induce_trainer import InducerTrainer

from utils.config import get_config_from_json
from utils.dirs import create_dirs
from utils.logger import DefinedSummarizer
from utils.utils import get_args

def process_config(json_file):
    config, _ = get_config_from_json(json_file)
    config.summary_dir = os.path.join(sys.path[0], "../experiments", config.exp_name, "summary/")
    config.checkpoint_dir = os.path.join(sys.path[0], "../experiments", config.exp_name, "checkpoint/")
    config.pseudo_data_dir = os.path.join(sys.path[0], "../experiments", config.exp_name, "pseudo_data_dir/")

    return config

def save_numpy(x_train, y_train, x_validate, y_validate, x_test, y_test, dir):
    with open(os.path.join(dir, "data_numpy.pkl"), 'wb') as f:
        pickle.dump({'x_train': x_train,
                     'y_train': y_train,
                     'x_validate': x_validate,
                     'y_validate': y_validate,
                     'x_test': x_test,
                     'y_test': y_test}, f)


def build_and_train_inducer(num_steps, config):
    # make tensorflow session
    sess = tf.Session()
    # create the experiments dirs
    create_dirs([config.summary_dir, config.checkpoint_dir, config.pseudo_data_dir])

    # get logger, data loader, model, builder, trainer
    logger = DefinedSummarizer(
        sess, summary_dir=config.summary_dir,
        scalar_tags=[
            'train/walk_loss',
            'train/fisher_loss',
            'train/acc',
            'train/acpt_rate'])

    data_loader = MnistDataLoader(config)

    model = MnistEnsembleModel(config)

    builder = InducerBuilder(data_loader, model, config)

    trainer = InducerTrainer(sess, data_loader, logger, config)

    # build graph
    builder.build()

    # train inducing points
    trainer.train(num_steps)

    x_pseudo, y_pseudo = sess.run(tf.get_collection('inputs_pseudo'))
    save_numpy(
        x_pseudo, y_pseudo,
        data_loader.x_validate, data_loader.y_validate,
        data_loader.x_test, data_loader.y_test,
        config.pseudo_data_dir)

    # close session
    sess.close()

    return x_pseudo, y_pseudo

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
    build_and_train_inducer(config.num_training_steps, config)

if __name__ == '__main__':
    main()

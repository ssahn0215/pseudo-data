import sys
import os

import tensorflow as tf
import numpy as np

sys.path.extend([os.path.join(sys.path[0],'..')])

from data_loader.mnist_loader import MnistDataLoader

from models.ensemble_model import MnistEnsembleModel

from builders.cla_builder import ClassifierBuilder

from trainers.cla_trainer import ClassifierTrainer

from utils.config import get_config_from_json
from utils.dirs import create_dirs
from utils.logger import DefinedSummarizer
from utils.utils import get_args

def process_config(json_file):
    config, _ = get_config_from_json(json_file)

    return config

def build_and_train_classifier(num_steps, config):
    # make tensorflow session
    sess = tf.Session()

    # get data loader, model, builder, trainer
    data_loader = MnistDataLoader(config)

    model = MnistEnsembleModel(config)

    builder = ClassifierBuilder(data_loader, model, config)

    trainer = ClassifierTrainer(sess, data_loader, None, config)

    # build graph
    builder.build()

    # train classifier
    trainer.train(num_steps)

    # test classifier
    loss, error = trainer.test()

    # close session
    sess.close()
    return loss, error

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
    build_and_train_classifier(config.num_training_steps, config)

if __name__ == '__main__':
    main()

import sys
import os

from easydict import EasyDict

import tensorflow as tf
import numpy as np

from induce_main import build_and_train_inducer
from cla_main import build_and_train_classifier

sys.path.extend([os.path.join(sys.path[0],'..')])

from utils.config import get_config_from_json
from utils.dirs import create_dirs
from utils.utils import get_args
from utils.plots import plot_images_and_labels

def process_config(json_file):
    config, _ = get_config_from_json(json_file)

    config.summary_dir = os.path.join(sys.path[0], "../experiments", config.exp_name, "summary/")
    config.img_dir = os.path.join(sys.path[0], "../experiments", config.exp_name, "images/")

    config_induce = EasyDict(config.config_induce)

    config_induce.exp_name = config.exp_name
    config_induce.summary_dir = config.summary_dir
    config_induce.checkpoint_dir = os.path.join(sys.path[0], "../experiments", config.exp_name, "checkpoint/")
    config_induce.pseudo_data_dir = os.path.join(sys.path[0], "../experiments", config.exp_name, "pseudo_data_dir/")

    config_cla = EasyDict(config.config_cla)

    config_cla.exp_name = config.exp_name
    config_cla.data_numpy_pkl = os.path.join(config_induce.pseudo_data_dir, "data_numpy.pkl")

    return config, config_induce, config_cla

def main():
    # capture the config path from the run arguments
    # then process the json configration file
    #try:
    args = get_args()
    #print(args)
    config, config_induce, config_cla = process_config(args.config)

    # set visible device
    os.environ["CUDA_VISIBLE_DEVICES"]=", ".join(args.gpu_name)

    # fix random seeds
    tf.set_random_seed(1)
    np.random.seed(1)
    tf.logging.set_verbosity(0)

    create_dirs([config.summary_dir, config.img_dir])

    # define external summary for classification
    cla_summary_writer = tf.summary.FileWriter(config.summary_dir)
    cla_pseudo_summary = tf.Summary()
    cla_loss_summary = tf.Summary()
    cla_error_summary = tf.Summary()

    # run induce function
    step, flag = 0, True
    while step < config_induce.num_training_steps:
        if flag:
            num_steps, flag = 0, False
        else:
            num_steps = min(
                config.num_steps_per_iter,
                config_induce.num_training_steps-step)

        print("Build and train inducer...")
        tf.reset_default_graph()
        x_pseudo, y_pseudo = build_and_train_inducer(num_steps, config_induce)

        step += num_steps

        plot_images_and_labels(x_pseudo, y_pseudo, 'pseudo_'+str(step), config.img_dir)

        print("Build and train classifier...")
        tf.reset_default_graph()
        loss, error = build_and_train_classifier(
            config_cla.num_training_steps, config_cla)

        cla_loss_summary.value.add(tag="test/loss", simple_value=loss)
        cla_error_summary.value.add(tag="test/error", simple_value=error)
        cla_summary_writer.add_summary(cla_loss_summary, step)
        cla_summary_writer.add_summary(cla_error_summary, step)
        cla_summary_writer.flush()

if __name__ == '__main__':
    main()

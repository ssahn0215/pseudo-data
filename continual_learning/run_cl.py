import numpy as np
import tensorflow as tf
import os
import sys
import protocols
sys.path.extend(['alg/'])
import models
import utils
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '-gpu','--gpu-name', nargs='+',
    default=['0','1','2','3'],
    help="name of gpu to use")
parser.add_argument(
    '-p', '--protocol', type=str,
    help="name of protocol to use")

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]=", ".join(args.gpu_name)
tf.reset_default_graph()
tf.set_random_seed(1)
np.random.seed(1)

# Network params
input_dim = 784
output_dim = 10
n_hidden_units = 100
model_size =[input_dim, n_hidden_units, n_hidden_units, output_dim]
#model_size =[input_dim, output_dim]
#[input_dim, n_hidden_units, n_hidden_units, output_dim]
nb_tasks = 10

# ## Construct dataset
full_dataset, final_test_dataset = utils.construct_permute_mnist(nb_tasks=nb_tasks)
training_dataset = full_dataset
validation_dataset = final_test_dataset

## Choose protocol
protocol_name, protocol = protocols.protocol_dict[args.protocol](model_size)

## Filename
datafile_name = "{}[size={}epochs={}].pkl.gz".format(
    protocol_name, str(model_size), protocol['epochs_per_task'])
print(datafile_name)

def run_fits(training_data, valid_data):
    evals = []
    tmp_evals = []
    model = protocol['model'](**protocol['model_args'])
    reg_model = protocol['reg_model'](
        target_params=model.params, **protocol['reg_model_args'])
    cl = protocol['cl_gen'](
        model=model,
        reg_model=reg_model,
        **protocol['cl_args'],
        data_size=training_data[0][0].shape[0])

    for age, task_idx in enumerate(range(nb_tasks)):
        print("Age is {}".format(age))
        X_train, y_train = training_data[task_idx]

        cl.pre_task_updates()
        cl.fit(X_train, y_train, protocol['epochs_per_task'])

        ftask = []
        for X_valid, y_valid in valid_data[:task_idx+1]:
            f_ = cl.evaluate(X_valid, y_valid)
            ftask.append(f_)

        print(np.mean(ftask))
        evals.append(ftask)

        next_task = True if task_idx != nb_tasks-1 else False
        cl.post_task_updates(X_train, y_train, next_task=next_task)

    return evals, tmp_evals

total_evals = []
evals, tmp_evals = run_fits(training_dataset, validation_dataset)

print([np.mean(e) for e in evals])
print([np.mean(e) for e in tmp_evals])

utils.save_zipped_pickle(evals, datafile_name)
evals = utils.load_zipped_pickle(datafile_name)

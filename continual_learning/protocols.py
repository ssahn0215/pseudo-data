import numpy as np
import tensorflow as tf
import os
import sys
sys.path.extend(['model/'])
import models
import reg_models
sys.path.extend(['alg/'])
import vanilla_cl
import fisher_cl
import mcmc_cl
import variational_cl
import gradient_replay_cl
sys.path.extend(['optimizer'])
import sgld

"""
A protocol is a function that takes as input some parameters and returns a tuple:
    (protocol_name, optimizer_kwargs)
The protocol name is just a string that describes the protocol.
The optimizer_kwargs is a dictionary that will get passed to KOOptimizer. It typically contains:
    step_updates, task_updates, task_metrics, regularizer_fn
"""

VANILLA_PROTOCOL = lambda model_size: (
    'vanilla',
    {
        'epochs_per_task': 20,
        'model': models.Deterministic_NN,
        'model_args': {
            'model_size': model_size,
            'batch_size': 256},
        'reg_model': reg_models.Gaussian_Regularizer,
        'reg_model_args': {},
        'cl_gen': vanilla_cl.Vanilla_CL,
        'cl_args': {
            'optimizer': tf.train.AdamOptimizer,
            'lam': 1.0,
            'lr_fun' : lambda global_step: 1e-3}
    }
    )

FISHER_PROTOCOL = lambda model_size: (
    'fisher',
    {
        'epochs_per_task': 10,
        'model': models.Deterministic_NN,
        'model_args': {
            'model_size': model_size,
            'batch_size': 256},
        'reg_model': reg_models.Gaussian_Regularizer,
        'reg_model_args': {},
        'cl_gen': fisher_cl.Fisher_CL,
        'cl_args': {
            'optimizer': tf.train.AdamOptimizer,
            'lam': 100.0,
            'lr_fun' : lambda global_step: 1e-3}
    }
    )

VARIATIONAL_PROTOCOL = lambda model_size: (
    'vcl',
    {
        'epochs_per_task': 100,
        'model': models.Stochastic_NN,
        'model_args': {
            'model_size': model_size,
            'batch_size': 256,
            'nb_ensembles': 10},
        'reg_model': reg_models.KL_Gaussian_Regularizer,
        'reg_model_args': {},
        'cl_gen': variational_cl.VCL_CL,
        'cl_args': {
            'optimizer': tf.train.AdamOptimizer,
            'lam': 1.0,
            'lr_fun' : lambda global_step: 1e-3,
            'nb_test_rep' : 10}
    }
    )

GAUSSIAN_MCMC_PROTOCOL = lambda model_size: (
    'g_mcmc',
    {
        'epochs_per_task': 10,
        'model': models.Deterministic_NN,
        'model_args': {
            'model_size': model_size,
            'batch_size': 256},
        'reg_model': reg_models.Gaussian_Score_Matching_Regularizer,
        'reg_model_args': {},
        'cl_gen': mcmc_cl.Gaussian_MCMC_CL,
        'cl_args': {
            'optimizer': sgld.StochasticGradientLangevinDynamics,
            'lam': 1.0,
            'lr_fun': lambda global_step: tf.train.exponential_decay(
                learning_rate=1e-3,
                global_step=global_step,
                decay_steps=1000,
                decay_rate=0.5,
                staircase=True),
            'burnin': 100,
            'thinning': 1}
    }
    )

GRADIENT_REPLAY_PROTOCOL = lambda model_size: (
    'gradient_replay',
    {
        'epochs_per_task': 20,
        'model': models.Deterministic_NN,
        'model_args': {
            'model_size': model_size,
            'batch_size': 256},
        'reg_model': reg_models.Gaussian_Regularizer,
        'reg_model_args': {},
        'cl_gen': gradient_replay_cl.Gradient_Replay_CL,
        'cl_args': {
            'optimizer': tf.train.AdamOptimizer,
            'lam': 0.01,
            'lr_fun' : lambda global_step: 1e-3}
    }
    )

FISHER_GAUSSIAN_SCORE_MATCHING_PROTOCOL = lambda model_size: (
    'fisher_gaussian_score_matching',
    {
        'epochs_per_task': 10,
        'model': models.Deterministic_NN,
        'model_args': {
            'model_size': model_size,
            'batch_size': 10000},
        'reg_model': reg_models.Gaussian_Regularizer,
        'reg_model_args': {},
        'cl_gen': fisher_cl.Fisher_Score_Matching_CL,
        'cl_args': {
            'optimizer': tf.train.AdamOptimizer,
            'lam': 1.0,
            'lr_fun' : lambda global_step: 1e-3}
    }
    )

protocol_dict = {
    'vanilla': VANILLA_PROTOCOL,
    'variational': VARIATIONAL_PROTOCOL,
    'g-mcmc': GAUSSIAN_MCMC_PROTOCOL,
    'fisher': FISHER_PROTOCOL,
    'gradient-replay': GRADIENT_REPLAY_PROTOCOL,
    'fgsm': FISHER_GAUSSIAN_SCORE_MATCHING_PROTOCOL}

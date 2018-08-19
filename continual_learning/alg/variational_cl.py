import tensorflow as tf
import numpy as np
import time
import utils
from vanilla_cl import Vanilla_CL

class VCL_CL(Vanilla_CL):
    def __init__(
        self,
        model,
        reg_model,
        optimizer,
        lam,
        lr_fun,
        data_size,
        nb_test_rep):

        self.nb_test_rep = nb_test_rep
        super(VCL_CL, self).__init__(
            model, reg_model, optimizer, lam, lr_fun, data_size)

    def fit_reg(self):
        params = self.model.get_params(self.sess)
        self.reg_model.assign_params(self.sess, params)

    def evaluate(self, X, y):
        tic = time.time()
        acc = utils.evaluate(
             self.sess, self.model, X, y,
             nb_rep=self.nb_test_rep)
        toc = time.time()
        print("Acc={:.3f}, time={:.2f}".format(acc, toc-tic))

        return acc

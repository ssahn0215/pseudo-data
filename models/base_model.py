import tensorflow as tf
import numpy as np

class BaseModel:
    def __init__(self, config):
        self.config = config

    def build_model(self):
        raise NotImplementedError

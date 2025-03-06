import torch
from abc import ABC

class DeepAgent(ABC):

    """ Abstract class for deep learning agents which share common methods """

    @staticmethod
    def encode_state(state):
        """ Turn the state into a tensor. """
        if state == ("terminal", "terminal"):
            state = (-1, -1)
        return torch.as_tensor(state, dtype=torch.float32)

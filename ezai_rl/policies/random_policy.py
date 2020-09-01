from . import BasePolicy
import warnings
import numpy as np

class RandomPolicy(BasePolicy):
    def __init__(self, action_space):
        super().__init__()
        self.action_space = action_space
        self.__name__ = 'Random'

    def get_action(self, *args, **kwargs):
        return self.action_space.sample()

    # No update function because the policy does not learn
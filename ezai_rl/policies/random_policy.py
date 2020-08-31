from . import BasePolicy
import numpy as np

class RandomPolicy(BasePolicy):
    def __init__(self, nA):
        super().__init__()
        self.nA = nA
        self.__name__ = 'Random'

    def get_action(self, *args, **kwargs):
        return np.random.choice(self.nA)
        #return env.action_space.sample()

    # No update function because the policy does not learn
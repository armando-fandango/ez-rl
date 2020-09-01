import warnings

class BasePolicy():
    def __init__(self):
        self.__name__="Not set"
        self.debug = False
        self.learn_mode = False

    def update(self, *args, **kwargs):
        warnings.warn("update() called on policy '{}' but it doesn't learn, hence no effect".format(self.__name__))

    def decay_er(self, *args, **kwargs):
        warnings.warn("decay_er() called on policy '{}' but it doesn't decay explore rate, hence no effect".format(self.__name__))

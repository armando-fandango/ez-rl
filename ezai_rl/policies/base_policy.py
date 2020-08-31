class BasePolicy():
    def __init__(self):
        self.__name__="Not set"
        self.debug = False
        self.learn_mode = False

    def update(self, *args, **kwargs):
        pass

    def decay_er(self, *args, **kwargs):
        pass
from abc import abstractmethod

class Layer:
    def __init__(self, name=None, lr=1e-3):
        if(name is not None):
            self.name = name
        else:
            self.name = "Unknown layer"

        self.lr = lr

    @abstractmethod
    def forward(self, x):
        pass

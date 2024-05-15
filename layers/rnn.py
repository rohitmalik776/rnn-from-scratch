import sys
import numpy as np

from .layer import Layer

sys.path.append("..")
from loss.cross_entropy_loss import cross_entropy_loss
from initializers.initializers import he_init, random_init

class RNN(Layer):
    def __init__(self, n_inp: int, n_hid: int, n_out: int, activation='tanh', name=None, lr=1e-3):
        Layer.__init__(self, name=name, lr=lr)
        self.activation = activation
        self.n_hid = n_hid
        self.n_inp = n_inp
        self.n_out = n_out

        init_fn = random_init

        # Hidden state
        self.h = np.zeros((n_hid, 1))

        # Hidden state weights
        self.Whh = init_fn((n_hid, n_hid))
        self.Whx = init_fn((n_hid, n_inp))
        self.bh = np.zeros((n_hid, 1))

        # Output weights
        self.Woh = init_fn((n_out, n_hid))
        self.bo = np.zeros((n_out, 1))

    def clear_state(self):
        self.h = np.zeros((self.n_hid, 1))
    
    # h(t) = g(Whh.h(t-1) + Whx.x(t) + bh)
    # y(t) = g(Wyh.h(t) + by)
    def step(self, x):
        h_t = np.dot(self.Whh, self.h) + np.dot(self.Whx, x) + self.bh
        h_t = np.tanh(h_t)
        self.h_t1 = self.h
        self.h = h_t

        y_t = np.dot(self.Woh, self.h) + self.bo
        y_t = np.tanh(y_t)
        
        return y_t

    def update(self, x, a, up_grad, batch_size):
        # dL / dZ
        dZ = np.multiply(up_grad, 1 - np.square(a))
        dWoh = np.dot(dZ, self.h.T)

        dbo = dZ

        dWhh = (np.dot(np.dot(self.Woh.T, dZ), self.h_t1.T))
        dWhx = np.dot(np.dot(self.Woh.T, dZ), x.T)

        dbh = np.dot(self.Woh.T, dZ)

        
        # Updation
        self.Woh = self.Woh - self.lr * dWoh
        self.Whh = self.Whh - self.lr * dWhh
        self.Whx = self.Whx - self.lr * dWhx

        self.bo = self.bo - self.lr * dbo
        self.bh = self.bh - self.lr * dbh

        # Downstream grad
        down_grad = np.dot(np.dot(self.Whx.T, self.Woh.T), dZ)
        return down_grad
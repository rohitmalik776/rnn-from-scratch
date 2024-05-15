import numpy as np

def cross_entropy_loss(y_hat, y):
    # Loss is always scalar
    epsilon = 1e-12
    y_hat = np.clip(y_hat, epsilon, 1. - epsilon)
    loss = -np.sum(y * np.log(y_hat))
    return loss

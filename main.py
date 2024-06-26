import time

import numpy as np

from layers.rnn import RNN
from loss.cross_entropy_loss import cross_entropy_loss

np.random.seed(0)

DS_PATH = './ds.txt'
# DS_PATH = './shakespeare.txt'
LEARNING_RATE = 1e-3
BATCH_SIZE = 16
T = 128

def load_ds(file_name):
    file = open(file_name)
    dsx = file.read()
    file.close()

    # Temporary clipping of dataset
    if(len(dsx) > 50000):
        dsx = dsx[:5000]

    dsy = [c for c in dsx]
    dsy.append(dsy.pop(0))
    return dsx, dsy

def preprocess(idx_to_char, char_to_idx, n_voc, ds):
    pds = []
    for x in ds:
        px = np.zeros((n_voc, 1))
        px[char_to_idx[x]][0] = 1
        pds.append(px)
    
    return pds

def batch(x, batch_size):
    batches = len(x) // batch_size
    x_batched = []
    for i in range(batches):
        batch = np.array(x[i * batch_size: (i+1) * batch_size]).reshape((-1, batch_size))
        x_batched.append(batch)
    return x_batched

# load dataset
dsx, dsy = load_ds(DS_PATH)
print("Dataset loaded!")
print("Size of x: ", len(dsx))

vocab = list(set(dsx))

# create lookup
idx_to_char = dict()
char_to_idx = dict()

for i in range(len(vocab)):
    char_to_idx[vocab[i]] = i
    idx_to_char[i] = vocab[i]

n_voc = len(vocab)

# preprocess dataset
pdsx = preprocess(idx_to_char, char_to_idx, n_voc, dsx)
pdsy = preprocess(idx_to_char, char_to_idx, n_voc, dsy)

# print(pdsx[0].shape)
# print(pdsx.shape)

print("Dataset pre-processed!")

rnn1 = RNN(name="rnn_1", n_inp=n_voc, n_out=100, n_hid=100, lr=LEARNING_RATE)
rnn2 = RNN(name="rnn_2", n_inp=100, n_out=n_voc, n_hid=100, lr=LEARNING_RATE)

start_time = time.time()

epochs = 10
for epoch in range(epochs):    
    rnn1.clear_state()
    rnn2.clear_state()
    total_loss = 0.0

    steps = 0
    grads1 = []
    grads2 = []

    for (x, y) in zip(pdsx, pdsy):
        # Forward propagation
        x1 = rnn1.step(x)
        a = rnn2.step(x1)

        loss = cross_entropy_loss(a, y)

        total_loss += loss
        steps += 1

        # Backpropagate
        # Just find the gradients here
        up_grad = -np.divide(y, a)
        dWoh, dWhh, dWhx, dbo, dbh, up_grad = rnn2.find_grads(a=a, x=x1, up_grad=up_grad, batch_size=BATCH_SIZE)
        grads2.append([dWoh, dWhh, dWhx, dbo, dbh])
        dWoh, dWhh, dWhx, dbo, dbh, up_grad = rnn1.find_grads(a=x1, x=x, up_grad=up_grad, batch_size=BATCH_SIZE)
        grads1.append([dWoh, dWhh, dWhx, dbo, dbh])

        # Update after T time steps
        if(len(grads1) == T):
            dWoh, dWhh, dWhx, dbo, dbh = np.zeros(shape=grads2[0][0].shape), np.zeros(shape=grads2[0][1].shape), np.zeros(shape=grads2[0][2].shape), np.zeros(shape=grads2[0][3].shape), np.zeros(shape=grads2[0][4].shape),  
            for(a, b, c, d, e) in grads2:
                dWoh += a
                dWhh += b
                dWhx += c
                dbo  += d
                dbh  += e
            
            rnn2.update(dWoh, dWhh, dWhx, dbo, dbh)

            dWoh, dWhh, dWhx, dbo, dbh = np.zeros(shape=grads1[0][0].shape), np.zeros(shape=grads1[0][1].shape), np.zeros(shape=grads1[0][2].shape), np.zeros(shape=grads1[0][3].shape), np.zeros(shape=grads1[0][4].shape),  
            for(a, b, c, d, e) in grads1:
                dWoh += a
                dWhh += b
                dWhx += c
                dbo  += d
                dbh  += e

            rnn1.update(dWoh, dWhh, dWhx, dbo, dbh)

            grads1.clear()
            grads2.clear()

    print("Epoch #" + str(epoch) + " loss: ", total_loss / steps)

end_time = time.time()
print("Time taken: ", end_time - start_time)

print("\nPredicting now!")

# Trying to predict only now
x = pdsx[0]
rnn1.clear_state()
rnn2.clear_state()
for i in range(50):
    x1 = rnn1.step(x)
    x = rnn2.step(x1)
    print(idx_to_char[np.argmax(x)], end='')
    # print(idx_to_char[np.argmax(y)])
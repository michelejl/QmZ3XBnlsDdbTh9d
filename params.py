import torch

#train-validation split
trainsplit = 0.8
valsplit = 1- trainsplit

# set batch_size
bs = 16

# set number of workers
num_workers = 2

#set the number of epochs
num_epochs=50

# early stopping patience; how long to wait after last time validation loss improved.
patience = 10

# set the device we will be using to train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
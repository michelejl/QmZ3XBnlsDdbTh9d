import torch

#train-validation split
trainsplit = 0.8
valsplit = 1- trainsplit

# set batch_size
bs = 16

# learning rate
lr=0.001

#set the number of epochs
num_epochs=1

# early stopping patience; how long to wait after last time validation loss improved.
patience = 7

# set the device we will be using to train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
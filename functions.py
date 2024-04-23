import torch # PyTorch package
import torch.nn as nn # basic building block for neural neteorks
import torch.optim as optim # optimzer
import matplotlib.pyplot as plt
import numpy as np
from params import num_epochs, bs
from pytorchtools import EarlyStopping

#Build the model
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            ## block 1
            nn.Conv2d(in_channels=3, out_channels=64,
                      kernel_size=(3, 3)),  ## Convo layer

            nn.ReLU(),                      ## ReLU activation function
            nn.MaxPool2d((2, 2)),           ## Pooling layer
            nn.Dropout(p=.1),               ## Dropout layer

            ## block 2
            nn.Conv2d(64, 32, (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Dropout(p=.1),
            
            ## block 3
            nn.Conv2d(32, 16, (2,2)),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Dropout(p=.1),
             
            nn.Flatten(),
            nn.Linear(10816, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64,1))
        
    def main_forward(self, x):
      return self.main(x)

    def sigmoid_activation(self, x):
      return torch.sigmoid(x)

    # Modified forward function
    def forward(self, x):
      main_output = self.main_forward(x)
      out = self.sigmoid_activation(main_output)
      return out
        
model = Net()

# initialize our optimizer and loss function
# specify loss function
lossFn = nn.BCELoss() 
# specify optimizer
opt = optim.Adam(model.parameters(), lr=0.001)



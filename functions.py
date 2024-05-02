import torch # PyTorch package
import torch.nn as nn # basic building block for neural neteorks
import torch.optim as optim # optimzer
import matplotlib.pyplot as plt
import numpy as np
from params import num_epochs, bs
from pytorchtools import EarlyStopping
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import Dataset

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
        
# Custom dataset class for handling sequences of images
class ImageSequenceDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence = self.data[idx]
        if self.transform:
            sequence = [self.transform(image) for image in sequence]
        return torch.stack(sequence)  # Stack images into a single tensor

# Define the model
class ImageSequenceClassifier(nn.Module):
    def __init__(self, num_classes, hidden_size):
        super(ImageSequenceClassifier, self).__init__()
        self.cnn = models.resnet18(pretrained=True)
        self.rnn = nn.LSTM(input_size=self.cnn.fc.in_features, 
                           hidden_size=hidden_size, 
                           num_layers=1, 
                           batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        batch_size, seq_length, channels, height, width = x.size()
        x = x.view(batch_size * seq_length, channels, height, width)
        features = self.cnn(x)
        features = features.view(batch_size, seq_length, -1)
        _, (h_n, _) = self.rnn(features)
        output = self.fc(h_n[-1])
        return output
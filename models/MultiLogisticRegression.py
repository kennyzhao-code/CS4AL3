import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

class MultiLogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super(MultiLogisticRegression, self).__init__() # to inherit the properties of the parent class nn.Module for pytorch
        # logistic regression model with a single layer (neural network with 1 layer)
        self.fc = nn.Linear(input_dim, 1) 

    def architecture(self, x):
        return torch.sigmoid(self.fc(x))  # sigmoid function, used commonly with logisticregresion


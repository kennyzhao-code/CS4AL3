import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

import torch
import torch.nn as nn
import torch.nn.functional as F

class ExoplanetNN(nn.Module):
    def __init__(self, input_size):
        super(ExoplanetNN, self).__init__()

        # Fully connected layers
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        
        # Output layer for tertiary classification
        self.output = nn.Linear(32, 3)
        
        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        # First hidden layer with ReLU activation and dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        # Second hidden layer with ReLU activation and dropout
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        # Third hidden layer with ReLU activation
        x = F.relu(self.fc3(x))
        
        # Output layer for class logits (no activation if using CrossEntropyLoss)
        x = self.output(x)

        return x
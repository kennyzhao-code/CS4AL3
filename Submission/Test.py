import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

class ExoplanetPreprocessor:
    def __init__(self, data_path):
        # Read the data
        self.df = pd.read_csv(data_path)
        
        # Identify column types
        self.categorical_columns = [
            'S_TYPE', 'P_TYPE'
        ]
        
        # Convert categorical columns to strings
        for col in self.categorical_columns:
            self.df[col] = self.df[col].astype(str)
        
        # Continuous columns (excluding categorical and label)
        self.continuous_columns = [
            col for col in self.df.columns 
            if col not in self.categorical_columns + ['Labels']
        ]
        
        # Preprocessor
        self.preprocessor = self._create_preprocessor()
        
    def _create_preprocessor(self):
        return ColumnTransformer(
            transformers=[
                # One-hot encoding for categorical variables
                ('cat', Pipeline([
                    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
                ]), self.categorical_columns),
                
                # Standard scaling for continuous variables
                ('num', Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]), self.continuous_columns)
            ])
    
    def preprocess(self):
        # Separate features and labels
        X = self.df.drop('Labels', axis=1)
        y = self.df['Labels']
        
        # Fit and transform the data
        X_processed = self.preprocessor.fit_transform(X)
        
        # Get feature names after preprocessing
        onehot_encoder = self.preprocessor.named_transformers_['cat'].named_steps['onehot']
        cat_feature_names = onehot_encoder.get_feature_names_out(self.categorical_columns)
        num_feature_names = self.continuous_columns
        feature_names = list(cat_feature_names) + list(num_feature_names)
        
        return X_processed, y, feature_names

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

def train_model(model, X_train, y_train, X_val, y_val, X_test, y_test, epochs=100, lr=0.001):
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        optimizer.zero_grad()
        outputs = model.forward(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        # Validation phase
        model.eval()
        with torch.no_grad():
            val_outputs = model.forward(X_val)
            val_loss = criterion(val_outputs, y_val)
            _, val_predicted = torch.max(val_outputs, 1)
            val_accuracy = (val_predicted == y_val).float().mean()

        # Print training and validation metrics
        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch+1}/{epochs}], "
                f"Training Loss: {loss.item():.4f}, "
                f"Validation Loss: {val_loss.item():.4f}, "
                f"Validation Accuracy: {val_accuracy.item():.4f}"
            )

    """
    This snippet of code contains the testing code for performing
    predictions on the test set and evaluating the model's performance.

    The previous code is required in order for this portion to run correctly.
    """

    # Final evaluation on the test set
    model.eval()
    with torch.no_grad():
        test_outputs = model.forward(X_test)
        _, test_predicted = torch.max(test_outputs, 1)
        test_accuracy = (test_predicted == y_test).float().mean()

    print(f"\nTest Accuracy: {test_accuracy.item():.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test.numpy(), test_predicted.numpy()))


# Preprocessing and data preparation
data_path = '../data/useddata/labeled_exoplanet_data.csv'
preprocessor = ExoplanetPreprocessor(data_path)
X_processed, y, feature_names = preprocessor.preprocess()

# Convert to PyTorch tensors
X_tensor = torch.tensor(X_processed, dtype=torch.float32)
y_tensor = torch.tensor(y.values, dtype=torch.long)

# Split the original dataset into 80% training and 20% test
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X_tensor, y_tensor, test_size=0.2, random_state=42
)

# Further split the training set into 80% training and 20% validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.2, random_state=42
)

# Initialize the model
model = ExoplanetNN(input_size=X_processed.shape[1])

# Print preprocessing information
print("Data preprocessed successfully!")
print(f"Number of features: {X_processed.shape[1]}")
print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")

# Train and evaluate the model
train_model(model, X_train, y_train, X_val, y_val, X_test, y_test)
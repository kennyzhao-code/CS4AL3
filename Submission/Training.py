import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import joblib
import itertools
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

''' Step 1: Load and assess the data '''
# check out any columns that look useless, columns of interest, data type of columns (continous, categorical, timeseries, etc.)

# turning the vot data to csv for GAIA
from astropy.io.votable import parse

votable = parse("./gaiaDR3HostStar.vot") # Load the VOTable file

table = votable.get_first_table().to_table() # Convert the first table in the VOTable to an Astropy Table

table.write("./gaiaDR3HostStar.csv", format="csv", overwrite=True) # save to csv

# Read the CSV files into a DataFrame
dfGAIA = pd.read_csv('./gaiaDR3HostStar.csv')
dfNASAExo = pd.read_csv('./NASA_Exoplanet_Archive.csv')
dfDropPHL = pd.read_csv('./PHL_Exoplanet_Habitability.csv')

''' Step 2: Clean and combine the data'''
columnsDropGAIA = [] # for Gaia, merge with sourceID 
columnsDropNASAExo = [
    "koi_disposition", "koi_pdisposition", "koi_score",  # Not related to habitability
    "koi_fpflag_nt", "koi_fpflag_ss", "koi_fpflag_co", "koi_fpflag_ec",  # Not related to habitability
    "koi_period_err1", "koi_period_err2",
    "koi_time0bk_err1", "koi_time0bk_err2",
    "koi_impact_err1", "koi_impact_err2",
    "koi_duration_err1", "koi_duration_err2",
    "koi_depth_err1", "koi_depth_err2",
    "koi_prad_err1", "koi_prad_err2",
    "koi_teq_err1", "koi_teq_err2",
    "koi_insol_err1", "koi_insol_err2",
    "koi_tce_plnt_num", "koi_tce_delivname",
    "koi_steff_err1", "koi_steff_err2",
    "koi_slogg_err1", "koi_slogg_err2",
    "koi_srad_err1", "koi_srad_err2",  # All above are error columns
    "kepid", "kepoi_name",  # High-cardinality identifiers
]
columnsDropPHL = [
    "P_RADIUS", "P_RADIUS_ERROR_MIN", "P_RADIUS_ERROR_MAX",
    "P_MASS_ERROR_MIN", "P_MASS_ERROR_MAX",  # Missing values in these columns
    "P_YEAR", "P_UPDATED",  # Not useful for habitability
    "P_PERIOD_ERROR_MIN", "P_PERIOD_ERROR_MAX",
    "P_SEMI_MAJOR_AXIS_ERROR_MIN", "P_SEMI_MAJOR_AXIS_ERROR_MAX",
    "P_ECCENTRICITY_ERROR_MIN", "P_ECCENTRICITY_ERROR_MAX",
    "P_INCLINATION_ERROR_MIN", "P_INCLINATION_ERROR_MAX",
    "P_OMEGA_ERROR_MIN", "P_OMEGA_ERROR_MAX",
    "P_TPERI_ERROR_MIN", "P_TPERI_ERROR_MAX",
    "P_IMPACT_PARAMETER", "P_IMPACT_PARAMETER_ERROR_MIN",
    "P_IMPACT_PARAMETER_ERROR_MAX", "P_TEMP_MEASURED",
    "P_GEO_ALBEDO", "P_GEO_ALBEDO_ERROR_MIN", "P_GEO_ALBEDO_ERROR_MAX",
    "P_DETECTION", "P_DETECTION_MASS", "P_DETECTION_RADIUS",
    "P_ALT_NAMES", "P_ATMOSPHERE",
    "S_DISTANCE_ERROR_MIN", "S_DISTANCE_ERROR_MAX",
    "S_METALLICITY_ERROR_MIN", "S_METALLICITY_ERROR_MAX",
    "S_MASS_ERROR_MIN", "S_MASS_ERROR_MAX",
    "S_RADIUS_ERROR_MIN", "S_RADIUS_ERROR_MAX",
    "S_AGE_ERROR_MIN", "S_AGE_ERROR_MAX",
    "S_TEMPERATURE_ERROR_MIN", "S_TEMPERATURE_ERROR_MAX",
    "S_DISC", "S_MAGNETIC_FIELD", "S_ALT_NAMES", "P_ESCAPE",
    "P_POTENTIAL", "P_GRAVITY", "P_DENSITY", "S_TYPE_TEMP",
    "P_TYPE_TEMP", "S_CONSTELLATION", "S_CONSTELLATION_ABR",
    "S_CONSTELLATION_ENG", "S_RA_H", "S_RA_T", "S_DEC_T", "S_NAME",
]

# dropping columns and recreating csv's
dfGAIA = dfGAIA.drop(columns=columnsDropGAIA)
dfNASAExo = dfNASAExo.drop(columns=columnsDropNASAExo)
dfDropPHL = dfDropPHL.drop(columns=columnsDropPHL)


dfGAIA.to_csv('./GAIA.csv', index=False)
dfNASAExo.to_csv('./NASAExo.csv', index=False)
dfDropPHL.to_csv('./PHL.csv', index=False)

'''Step 3: Merge NASAExo and PHL'''

# Load preprocessed datasets
dfNASAExo = pd.read_csv('./NASAExo.csv')
dfPHL = pd.read_csv('./PHL.csv')

# Rename columns for consistency
dfPHL.rename(columns={"P_NAME": "planet_name", "S_NAME": "star_name"}, inplace=True)
dfNASAExo.rename(columns={"kepler_name": "planet_name"}, inplace=True)

# Perform merges
finalMerge = pd.merge(dfNASAExo, dfPHL, on="planet_name", how="right")

# Fill empty values with a placeholder (e.g., 0)
finalMerge.fillna(0, inplace=True)

# Save merged dataset
finalMerge.to_csv('./merged_NASAExo_PHL.csv', index=False)

# Drop planet_name
finalMerge = finalMerge.drop(columns=["planet_name"])

# Fill empty values again if necessary after dropping columns
finalMerge.fillna(0, inplace=True)

# Save the final cleaned dataset
finalMerge.to_csv('./final_NASAExo_PHL.csv', index=False)

def assign_labels(data):
    labels = []
    for _, row in data.iterrows():
        # Extract relevant features
        P_RADIUS_EST = row.get('P_RADIUS_EST', None)
        P_TEMP_EQUIL = row.get('P_TEMP_EQUIL', None)
        P_FLUX = row.get('P_FLUX', None)
        P_PERIOD = row.get('P_PERIOD', None)
        P_MASS_EST = row.get('P_MASS_EST', None)
        # S_RADIUS = row.get('S_RADIUS', None)

        # Check for missing data
        criteria = [
            (0.3 <= P_RADIUS_EST <= 10) if P_RADIUS_EST is not None else False,
            (100 <= P_TEMP_EQUIL <= 450) if P_TEMP_EQUIL is not None else False,
            (0.01 <= P_FLUX <= 4) if P_FLUX is not None else False,
            (5 <= P_PERIOD <= 2000) if P_PERIOD is not None else False,
            (0.1 <= P_MASS_EST <= 20) if P_MASS_EST is not None else False,
            # (0.5 <= S_RADIUS <= 2.0) if S_RADIUS is not None else False,
        ]

        # Habitable: At least 4 of the criteria must be satisfied
        if sum(criteria) >= 4:
            labels.append(1)  # Habitable

        # Not Habitable: Fewer than 3 criteria satisfied and no extreme outliers
        elif sum(criteria) < 3:
            labels.append(0)  # Not Habitable

        # Default to Unknown for edge cases
        else:
            labels.append(2)  # Unknown

    data['Labels'] = labels
    return data

data_path = './final_NASAExo_PHL.csv'
data = pd.read_csv(data_path)
data_with_labels = assign_labels(data)
data_with_labels.to_csv('./labeled_exoplanet_datatestwsradcsv.csv', index=False)

# Visualize label distribution
labels_count = data_with_labels['Labels'].value_counts()
labels_count.plot(kind='bar', title='Label Distribution')
plt.xlabel('Labels')
plt.ylabel('Count')
plt.show()

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
        self.fc4 = nn.Linear(32, 16)

        # Output layer for tertiary classification
        self.output = nn.Linear(16, 3)
        
        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # First hidden layer with ReLU activation and dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        # Second hidden layer with ReLU activation and dropout
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        # Third hidden layer with ReLU activation
        x = F.relu(self.fc3(x))
        x = self.dropout(x)

        # Fourth hidden layer with ReLU activation
        x = F.relu(self.fc4(x))
        x = self.dropout(x)
        
        # Output layer for class logits (no activation if using CrossEntropyLoss)
        x = self.output(x)

        return x
    
# Multi logistic regression model used for comparison reasons
class ExoplanetMLR(nn.Module):
    def __init__(self, input_dim):
        super(ExoplanetMLR, self).__init__() # to inherit the properties of the parent class nn.Module for pytorch
        # logistic regression model with a single layer (neural network with 1 layer)
        self.fc = nn.Linear(input_dim, 1) 

    def architecture(self, x):
        return torch.sigmoid(self.fc(x))  # sigmoid function, used commonly with logisticregresion


def train_model_MLR(model, X_train, y_train, X_val, y_val, X_test, y_test, epochs=300, lr=0.001, patience=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0

    # Lists to store loss and accuracy values for plotting
    train_losses = []
    val_losses = []
    val_accuracies = []

    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        optimizer.zero_grad()
        outputs = model.forward(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        # Validation phase
        model.eval()
        with torch.no_grad():
            val_outputs = model.forward(X_val)
            val_loss = criterion(val_outputs, y_val)
            _, val_predicted = torch.max(val_outputs, 1)
            val_accuracy = (val_predicted == y_val).float().mean()
            val_losses.append(val_loss.item())
            val_accuracies.append(val_accuracy.item())

        # Early stopping
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            patience_counter = 0

            # Save the best model
            best_model_state = model.state_dict()
        else:
            # Increment counter if no improvement
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                break

        # Print training and validation metrics
        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch + 1}/{epochs}], "
                f"Training Loss: {loss.item():.4f}, "
                f"Validation Loss: {val_loss.item():.4f}, "
                f"Validation Accuracy: {val_accuracy.item():.4f}"
            )

    # Restore the best model state
    model.load_state_dict(best_model_state)

    # Final evaluation on the test set
    model.eval()
    with torch.no_grad():
        test_outputs = model.forward(X_test)
        _, test_predicted = torch.max(test_outputs, 1)
        test_accuracy = (test_predicted == y_test).float().mean()

    print(f"\nTest Accuracy: {test_accuracy.item() * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test.numpy(), test_predicted.numpy()))

    # Plotting loss curves
    actual_epochs = len(train_losses)
    plt.figure(figsize=(12, 6))

    # Plot training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(range(1, actual_epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, actual_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()

    # Plot validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(1, actual_epochs + 1), val_accuracies, label='Validation Accuracy', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy Curve')
    plt.legend()

    plt.tight_layout()
    plt.show()

def train_model(model, X_train, y_train, X_val, y_val, X_test, y_test, epochs=300, lr=0.001, patience=10):
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # Early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0

    # Lists to store loss and accuracy values for plotting
    train_losses = []
    val_losses = []
    val_accuracies = []

    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        optimizer.zero_grad()
        outputs = model.forward(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        # Validation phase
        model.eval()
        with torch.no_grad():
            val_outputs = model.forward(X_val)
            val_loss = criterion(val_outputs, y_val)
            _, val_predicted = torch.max(val_outputs, 1)
            val_accuracy = (val_predicted == y_val).float().mean()
            val_losses.append(val_loss.item())
            val_accuracies.append(val_accuracy.item())

        # Early stopping
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            patience_counter = 0

            # Save the best model
            best_model_state = model.state_dict()
        else:
            # Increment counter if no improvement
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                break

        # Print training and validation metrics
        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch + 1}/{epochs}], "
                f"Training Loss: {loss.item():.4f}, "
                f"Validation Loss: {val_loss.item():.4f}, "
                f"Validation Accuracy: {val_accuracy.item():.4f}"
            )

    # Restore the best model state
    model.load_state_dict(best_model_state)

    # Final evaluation on the test set
    model.eval()
    with torch.no_grad():
        test_outputs = model.forward(X_test)
        _, test_predicted = torch.max(test_outputs, 1)
        test_accuracy = (test_predicted == y_test).float().mean()

    print(f"\nTest Accuracy: {test_accuracy.item() * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test.numpy(), test_predicted.numpy()))

    # Plotting loss curves
    actual_epochs = len(train_losses)
    plt.figure(figsize=(12, 6))

    # Plot training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(range(1, actual_epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, actual_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()

    # Plot validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(1, actual_epochs + 1), val_accuracies, label='Validation Accuracy', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy Curve')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Preprocessing and data preparation
data_path = './labeled_exoplanet_datatestwsradcsv.csv'
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

# Define hyperparameters to test
learning_rates = [0.0001, 0.001, 0.01, 0.1]
epochs_list = [50, 100, 200, 300]

# # Test the hyperparameters
# results = test_hyperparameters(ExoplanetNN(input_size=X_processed.shape[1]), 
#                                X_train, y_train, X_val, y_val, X_test, y_test, 
#                                learning_rates, epochs_list)

# Initialize the model
model = ExoplanetNN(input_size=X_processed.shape[1])

# Print preprocessing information
print("Data preprocessed successfully!")
print(f"Number of features: {X_processed.shape[1]}")
print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")

### Found that learning rate = 0.001 and epochs = 300 give the best results ###

# Train and evaluate the model
train_model(model, X_train, y_train, X_val, y_val, X_test, y_test, epochs=200, lr=0.001)

torch.save(model.state_dict(), 'model.pkl')
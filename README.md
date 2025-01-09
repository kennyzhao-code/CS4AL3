# Exoplanet Habitability Classifier

This project explores the application of machine learning to predict the habitability of exoplanets. With over 5,600 confirmed exoplanets and billions more uncharted, the aim is to identify potentially habitable candidates using neural networks. By analyzing orbital, planetary, and stellar data, the model categorizes exoplanets as habitable, not habitable, or unknown, offering insights for future astronomical exploration.

## Table of Contents

- [Problem Context](#problem-context)
- [Features](#features)
- [Datasets](#datasets)
- [Model Details](#model-details)
  - [Architecture](#architecture)
  - [Improvements](#improvements)
  - [Preprocessing](#preprocessing)
- [Results](#results)
  - [Preliminary Results (Milestone 2)](#preliminary-results-milestone-2)
  - [Final Results (Milestone 3)](#final-results-milestone-3)
- [Limitations](#limitations)
- [Getting Started](#getting-started)
- [Additional References](#additional-references)

## Problem Context

As Earth's resources dwindle, the need to explore beyond our solar system becomes urgent. Exoplanets are too distant for direct exploration, making machine learning an efficient tool for classifying their habitability. This project aims to streamline this process, contributing to humanity's understanding of potentially habitable worlds.

## Features

- **Input Variables**: Planetary, orbital, and stellar attributes, transit-based features, and habitability metrics.
- **Labels**:
  - 0: Not Habitable
  - 1: Habitable
  - 2: Unknown
- **ML Technique**: Neural networks for non-linear relationships and feature interactions.

## Datasets

1. **NASA Exoplanet Archive**:

   - Orbital and planetary properties.
   - 49 features after preprocessing.

2. **PHL Habitable Zone Catalog**:

   - Habitability labels.
   - 118 features after preprocessing.

3. **Gaia DR3 Subset**:
   - Host star properties.
   - 8 features after preprocessing.

Data preprocessing included merging datasets, handling NA values, feature selection, and scaling. Redundant datasets were removed for efficiency, and features were aligned across sources for consistency.

## Model Details

### Architecture

- **Type**: Neural Network
- **Layers**: 4
- **Activation Function**: ReLU
- **Optimizer**: Adam
- **Loss Function**: Cross-Entropy Loss
- **Regularization**: Early stopping
- **Training Parameters**:
  - Batch size: 64
  - Learning rate: 0.001
  - Epochs: 200

### Improvements

- **Labeling Strategy**: Adjusted to ensure a balanced distribution across classes (habitable, non-habitable, unknown).
- **Optimizer**: Switched from Stochastic Gradient Descent (SGD) to Adam for improved handling of high-dimensional data.
- **Model Complexity**: Increased depth to 4 layers for improved accuracy without overfitting.
- **Regularization**: Implemented early stopping to prevent overfitting during training.

### Preprocessing

- Removed redundant features and merged datasets.
- Normalized continuous variables using StandardScaler.
- Applied one-hot encoding for categorical features.
- Split data into 80% training and 20% testing sets.

## Results

- **Accuracy**: Achieved ~91% accuracy with optimized parameters.
- **Validation Strategy**: K-fold cross-validation with metrics including accuracy, precision, and recall.
- **Model Comparison**: Outperformed logistic regression, achieving better results with fewer epochs.

### Preliminary Results (Milestone 2)

- Observed overfitting due to biased labeling and lack of regularization.
- Validation accuracy plateaued at 100%, indicating issues with preprocessing and labeling strategies.

### Final Results (Milestone 3)

- Revised labeling strategy reduced bias and improved distribution of classes.
- Adam optimizer and additional hidden layers increased accuracy to ~91%.
- Comparison with logistic regression showed superior efficiency and accuracy.

## Limitations

- The accuracy of predictions is sensitive to labeling strategies, which required iterative refinement.
- Early iterations of the model exhibited overfitting, highlighting the importance of regularization techniques.
- Performance is limited by the quality and completeness of the input data.

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/kennyzhao-code/Exoplanet-Habitability-Classifier.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the training script located under `Submission_3`:
   ```bash
   python Training.py
   ```
4. Run the test script located under `Submission_3`:
   ```bash
   python Test.py
   ```

## Additional References

- [NASA Exoplanets](https://science.nasa.gov/exoplanets/)
- [Adam Optimizer - GeeksforGeeks](https://www.geeksforgeeks.org/adam-optimizer/)
- [Scikit-learn StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)

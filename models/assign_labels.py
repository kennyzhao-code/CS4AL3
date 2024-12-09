import pandas as pd
import os

# Define the base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Paths for input and output files
INPUT_FILE = os.path.join(BASE_DIR, 'data', 'mergeddata', 'merged_NASAExo_PHL.csv')
OUTPUT_FILE = os.path.join(BASE_DIR, 'data', 'useddata', 'labeled_exoplanet_data.csv')

# Function to assign labels
def assign_labels(row):
    # Check for missing critical features
    if pd.isnull(row['koi_prad']) or pd.isnull(row['koi_teq']) or pd.isnull(row['P_FLUX']):
        return 2  # Unknown

    # Habitable criteria
    if 0.5 <= row['koi_prad'] <= 2 and 200 <= row['koi_teq'] <= 350:
        return 0  # Habitable

    # Not habitable criteria
    if row['koi_prad'] > 2 or row['koi_teq'] < 200 or row['koi_teq'] > 350:
        return 1  # Not Habitable

    # Default to Unknown
    return 2

# Main function to process the dataset
def process_dataset():
    # Load the dataset
    print("Loading dataset...")
    try:
        data = pd.read_csv(INPUT_FILE)
        print(f"Dataset loaded. Number of rows: {len(data)}")
    except FileNotFoundError:
        print(f"Error: Input file not found at {INPUT_FILE}")
        return

    # Apply the labeling function
    print("Assigning labels...")
    data['labels'] = data.apply(assign_labels, axis=1)

    # Save the labeled dataset
    print(f"Saving labeled dataset to {OUTPUT_FILE}...")
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)  # Ensure the output directory exists
    data.to_csv(OUTPUT_FILE, index=False)
    print("Labeled dataset saved successfully.")

# Run the script
if __name__ == '__main__':
    process_dataset()

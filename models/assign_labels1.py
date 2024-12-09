import pandas as pd

def assign_complex_labels(data):
    labels = []
    for _, row in data.iterrows():
        # Extract relevant features
        koi_prad = row['koi_prad']
        koi_teq = row['koi_teq']
        koi_steff = row['koi_steff']
        koi_srad = row['koi_srad']
        koi_period = row['koi_period']
        koi_impact = row['koi_impact']
        koi_slogg = row['koi_slogg']
        
        # Check for missing data
        if any(pd.isna([koi_prad, koi_teq, koi_steff, koi_srad, koi_period, koi_impact, koi_slogg])):
            labels.append(2)  # 2 represents Unknown
            continue
        
        # Habitable criteria
        if (0.5 <= koi_prad <= 2 and
            200 <= koi_teq <= 350 and
            4000 <= koi_steff <= 7000 and
            0.8 <= koi_srad <= 1.5 and
            50 <= koi_period <= 500 and
            koi_impact <= 1 and
            koi_slogg >= 4.0):
            labels.append(0)  # 0 represents Habitable
        else:
            # Non-Habitable criteria
            labels.append(1)  # 1 represents Not Habitable
    
    data['Labels'] = labels
    return data

# Usage
data_path = 'data/mergeddata/merged_NASAExo_PHL.csv'
data = pd.read_csv(data_path)
data_with_labels = assign_complex_labels(data)
data_with_labels.to_csv('data/useddata/labeled_exoplanet_data1.csv', index=False)

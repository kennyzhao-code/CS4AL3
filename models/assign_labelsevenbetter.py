import pandas as pd

def assign_labels(data):
    labels = []
    for _, row in data.iterrows():
        # Extract relevant features
        koi_prad = row['koi_prad']
        koi_teq = row['koi_teq']
        P_FLUX = row['P_FLUX']
        
        # Check for missing data
        if any(pd.isna([koi_prad, koi_teq, P_FLUX])):
            labels.append(2)  # Unknown due to missing data
            continue

        # Habitable criteria (very relaxed)
        if (0.3 <= koi_prad <= 20 and  # Includes sub-Earth to Jupiter-sized planets
            10 <= koi_teq <= 800 and   # Very broad temperature range
            0.001 <= P_FLUX <= 20):    # Very wide flux range
            labels.append(1)  # Habitable

        # Unknown criteria (edge cases)
        elif (20 < koi_prad <= 30 or  # Very large planets that could host moons
              800 < koi_teq <= 1000 or  # Extremely hot planets with potential atmospheres
              0.0001 <= P_FLUX < 0.001 or 20 < P_FLUX <= 30):  # Marginal flux values
            labels.append(2)  # Unknown

        # Not Habitable criteria
        elif (koi_prad > 30 or  # Extremely large planets
              koi_teq < 10 or koi_teq > 1000 or  # Extreme temperatures
              P_FLUX < 0.0001 or P_FLUX > 30):   # Extremely low or high flux
            labels.append(0)  # Not Habitable

        # Default to Unknown for safety
        else:
            labels.append(2)

    data['Labels'] = labels
    return data

data_path = '../data/mergeddata/final_NASAExo_PHL.csv'
data = pd.read_csv(data_path)
data_with_labels = assign_labels(data)
data_with_labels.to_csv('../data/useddata/labeled_exoplanet_datatestevenbetter.csv', index=False)
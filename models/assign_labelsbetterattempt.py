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

        # Habitable criteria (significantly relaxed)
        if (0.5 <= koi_prad <= 15 and  # Includes large planets up to Jupiter size
            50 <= koi_teq <= 600 and   # Wide temperature range
            0.01 <= P_FLUX <= 10):     # Very broad flux range
            labels.append(1)  # Habitable

        # Unknown criteria (clear edge cases)
        elif (15 < koi_prad <= 20 or  # Very large planets
              600 < koi_teq <= 800 or  # Hotter planets with potential dense atmospheres
              0.001 <= P_FLUX < 0.01 or 10 < P_FLUX <= 15):  # Marginal flux values
            labels.append(2)  # Unknown

        # Not Habitable criteria
        elif (koi_prad > 20 or  # Extremely large planets
              koi_teq < 50 or koi_teq > 800 or  # Very extreme temperatures
              P_FLUX < 0.001 or P_FLUX > 15):   # Extremely low or high flux
            labels.append(0)  # Not Habitable

        # Default to Unknown for safety
        else:
            labels.append(2)

    data['Labels'] = labels
    return data

data_path = '../data/mergeddata/final_NASAExo_PHL.csv'
data = pd.read_csv(data_path)
data_with_labels = assign_labels(data)
data_with_labels.to_csv('../data/useddata/labeled_exoplanet_datatestba1.csv', index=False)
import pandas as pd

def assign_labels(data):
    labels = []
    for _, row in data.iterrows():
        # Extract relevant features
        koi_prad = row['koi_prad']
        koi_teq = row['koi_teq']
        p_flux = row['P_FLUX']
        
        # Check for missing data
        if pd.isnull(koi_prad) or pd.isnull(koi_teq) or pd.isnull(p_flux):
            labels.append(2)  # 2 represents Unknown
            continue
        
        # Habitable criteria
        if 0.5 <= koi_prad <= 2 and 200 <= koi_teq <= 350:
            labels.append(1)  # 0 represents Habitable
        # Not Habitable criteria
        elif koi_prad > 2 or koi_teq < 200 or koi_teq > 350:
            labels.append(0)  # 1 represents Not Habitable
        else:
            # Default to Unknown if no clear classification
            labels.append(2)  # 2 represents Unknown
    
    data['Labels'] = labels
    return data
    
data_path = '../data/mergeddata/final_NASAExo_PHL.csv'
data = pd.read_csv(data_path)
data_with_labels = assign_labels(data)
data_with_labels.to_csv('../data/useddata/labeled_exoplanet_datatest3.csv', index=False)

    
    
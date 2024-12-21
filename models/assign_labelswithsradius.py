import pandas as pd

def assign_labels(data):
    labels = []
    for _, row in data.iterrows():
        # Extract relevant features
        P_RADIUS_EST = row.get('P_RADIUS_EST', None)
        P_TEMP_EQUIL = row.get('P_TEMP_EQUIL', None)
        P_FLUX = row.get('P_FLUX', None)
        P_PERIOD = row.get('P_PERIOD', None)
        P_MASS_EST = row.get('P_MASS_EST', None)
        S_RADIUS = row.get('S_RADIUS', None)

        # Check for missing data
        criteria = [
            (0.3 <= P_RADIUS_EST <= 10) if P_RADIUS_EST is not None else False,
            (100 <= P_TEMP_EQUIL <= 450) if P_TEMP_EQUIL is not None else False,
            (0.01 <= P_FLUX <= 4) if P_FLUX is not None else False,
            (5 <= P_PERIOD <= 2000) if P_PERIOD is not None else False,
            (0.1 <= P_MASS_EST <= 20) if P_MASS_EST is not None else False,
            (0.5 <= S_RADIUS <= 2.0) if S_RADIUS is not None else False,
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

data_path = '../data/mergeddata/final_NASAExo_PHL.csv'
data = pd.read_csv(data_path)
data_with_labels = assign_labels(data)
data_with_labels.to_csv('../data/useddata/labeled_exoplanet_datatestwsradcsv', index=False)
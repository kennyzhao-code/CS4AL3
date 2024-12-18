import pandas as pd

def assign_complex_labels(data):
    labels = []
    for _, row in data.iterrows():
        # Extract relevant features
        radius = row["koi_prad"]
        temp = row["koi_teq"]
        flux = row["koi_insol"]
        semi_major_axis = row["P_SEMI_MAJOR_AXIS"]
        hz_min = row["S_HZ_OPT_MIN"]
        hz_max = row["S_HZ_OPT_MAX"]
        p_type = row["P_TYPE"]

        # Check for missing data
        if any(pd.isna([radius, temp, flux, hz_min, hz_max, semi_major_axis, p_type])):
            labels.append(2)  # 2 represents Unknown
            continue

        # Habitable criteria (relaxed)
        if (0.5 <= radius <= 3.0 and  # Include larger rocky planets and small gas planets
            200 <= temp <= 400 and    # Allow colder and slightly hotter temperatures
            0.1 <= flux <= 2.0 and    # Broaden acceptable stellar flux
            (hz_min <= semi_major_axis <= hz_max) and  # Still within the star's habitable zone
            p_type in ["Terran", "Sub-Terran", "Super-Terran"]):  # Include similar planet types
            labels.append(1)  # 1 represents Habitable
        else:
            # Non-Habitable criteria
            labels.append(0)  # 0 represents Not Habitable

    data['labels'] = labels
    return data

data_path = '../data/mergeddata/final_NASAExo_PHL.csv'
data = pd.read_csv(data_path)
data_with_labels = assign_complex_labels(data)
data_with_labels.to_csv('../data/useddata/labeled_exoplanet_datatest2.csv', index=False)
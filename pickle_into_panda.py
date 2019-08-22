import pickle
import numpy
import pandas as pd

with open("data.pkl", "rb") as f:
    ad_ages = pickle.load(f)
    ad_ct_cs = pickle.load(f)
    ad_ct_long = pickle.load(f)
    hc_ages = pickle.load(f)
    hc_ct_cs = pickle.load(f)
    hc_ct_long = pickle.load(f)

# Create AD DataFrame, combining the cross-sectional regional CT data, ages, and a group label
ad_df = pd.DataFrame(data=ad_ct_cs)
ad_df['age'] = ad_ages
ad_df['AD'] = 1

# Create HC DataFrame, combining the cross-sectional regional CT data, ages, and a group label
hc_df = pd.DataFrame(data=hc_ct_cs)
hc_df['age'] = hc_ages
hc_df['AD'] = 0

# Append one list to another
data = ad_df.append(hc_df)

print(data.head())
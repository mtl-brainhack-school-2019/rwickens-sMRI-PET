import pickle
import numpy
import pandas as pd

with open("data.pkl", "rb") as f:
    ad_ages = pickle.load(f) #ages AD
    ad_ct_cs = pickle.load(f) #cortical thickness AD
    ad_ct_long = pickle.load(f) #longitudinal data AD
    hc_ages = pickle.load(f) #ages healthy controls
    hc_ct_cs = pickle.load(f) #cortical thickness healthy controls
    hc_ct_long = pickle.load(f) #longitudinal data healthy controls

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

#let's get a sense of the number of subjects in each group. 

print(ad_df.shape)
print(hc_df.shape)

#data.to_csv(r'C:\Users\rwick\Documents\GitHubNew\rwickens-sMRI-PET\cortical_data.csv')

#Note to self: 
#indices 0 to 999 are AD 
#indices 1000 to 1999 are HC

ages = data['age']
print(ages)

diagnosis = data['AD']
print(diagnosis)

thickness = data[0:-1]
print(thickness)
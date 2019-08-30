"""
Using synthetic data graciously created by Joshua Morse, I am attempting to create one or more
classifiers to predict group diagnosis - Alzheimer's disease (1) versus healthy control (0)

Ideas for classifiers to explore:
- (Gaussian) naive bayes 
- Logistic regression 
- Discriminant function analysis
- Random forest 

Goals: 
- Run k-s tests on predictor variables to check assumptions
- Create ROCs
- Look into implementing regularizers / penalty functions / sparsity
"""

import pickle
import numpy as np
import pandas as pd
import sklearn
from sklearn import metrics
from sklearn.model_selection import train_test_split
#from sklearn.model_selection import KFold
#from sklearn.model_selection import cross_val_score # possibly for continuous data only
from sklearn.naive_bayes import GaussianNB
import scipy.stats as stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification


# GETTING X AND Y VARIABLES READY ##################

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

#Note to self: indices 0-999 are AD; indices 1000 to 1999 are HC

ages = data['age']
#print(ages)

diagnosis = data['AD']
#print(diagnosis)

thickness = data.iloc[:,0:-2]
#print(thickness)

predictors = data.iloc[:,0:-1]

#####################

# split the data into train and test sets (75/25), also indicate stratification of diagnosis
xtrain, xtest, ytrain, ytest = train_test_split(predictors, diagnosis, stratify = diagnosis, random_state=1)

#check underlying assumption before running Gaussian naive bayes classifier 

#Pseudocode:
# turn each column of test DF into numpy array
# cycle through and run stats.ktest(x[i], 'norm')
# append results into some list with printed index
# if the proportion of significant results is < than my alpha level (0.05), I'll consider this a safe endeavour 
 
# x = numpy array
# stats.kstest(x, 'norm')

model = GaussianNB()
model.fit(xtrain, ytrain)
y_hat = model.predict(xtest)
#yprob_test = model.predict_proba(xtest)

# plot prior probabilities 











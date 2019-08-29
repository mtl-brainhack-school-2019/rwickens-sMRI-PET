#!/usr/bin/env python
# coding: utf-8

import warnings
warnings.filterwarnings("ignore")
from nilearn import datasets
from nilearn import plotting
from nilearn.plotting import plot_stat_map, show
import copy
from copy import deepcopy
import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from nilearn.input_data import NiftiMasker
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression
from sklearn.svm import SVR
from sklearn.decomposition import PCA
#from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

"""
#**********************************************************************
#*Only run this chunk of code once to download arrays to disk! 

subjects = 403
oasis_dataset = datasets.fetch_oasis_vbm(n_subjects = subjects) #type is sklearn.utils.bunch

#Let's plot the first subject's brain scan for fun. 
subject1 = oasis_dataset.gray_matter_maps[0]
plotting.view_img(subject1)

with open("gray_filenames.pkl", "ab") as f:
    pickle.dump(oasis_dataset.gray_matter_maps, f)

nifti_masker = NiftiMasker(
    standardize=False,
    smoothing_fwhm=2,
    memory='nilearn_cache')  # cache options

gm_maps_masked_nondisk = nifti_masker.fit_transform(oasis_dataset.gray_matter_maps)
age_nondisk = oasis_dataset.ext_vars['age'].astype(float)

#Save grey matter array onto disk
np.save("C:/Users/rwick/Documents/GitHub/rwickens-sMRI-PET/oasis.npy", gm_maps_masked_nondisk)

#Save age (our y-variable) array onto disk 
np.save("C:/Users/rwick/Documents/GitHub/rwickens-sMRI-PET/ages.npy", age_nondisk)

#*****************************************************
"""

with open("gray_filenames.pkl", "rb") as f:
    gray_matter_map_filenames = pickle.load(f)

#load grey matter array (x-variables) array from disk
gm_maps_masked = np.load("C:/Users/rwick/Documents/GitHubNew/rwickens-sMRI-PET/oasis.npy")

#Load age array (y-variable) from disk
age = np.load("C:/Users/rwick/Documents/GitHubNew/rwickens-sMRI-PET/ages.npy")

# Locate these files on your disk
print('First gray-matter anatomy image (3D) is located at: %s' %
      gray_matter_map_filenames[0])  # 3D data
#print('First white-matter anatomy image (3D) is located at: %s' %
      #oasis_dataset.white_matter_maps[0])  # 3D data

#Let's plot the first subject's brain scan for fun. 
subject1 = gray_matter_map_filenames[0]
plotting.view_img(subject1)
#save this image somewhere

n_samples, n_features = gm_maps_masked.shape
print(age.shape)
print("%d samples, %d features" % (n_samples, n_features))

print(np.amax(gm_maps_masked))
#Hm. I wonder what units this is in. I was under the impression
#that tissue values represent probability, which should be between 0 and 1.

print(np.amin(gm_maps_masked))
#A minimum of 0 is good. 

#Now, here is my larger model (dimension reduction and prediction models) 

print("PCA + SVR")
GMtrain, GMtest, age_train, age_test = train_test_split(gm_maps_masked, age, random_state=1)

# Define the prediction function to be used.
# Here we use a Support Vector Classification, with a linear kernel
svr = SVR(kernel='linear')

# Dimension reduction - using a PCA with the first 100 components
# Making sure to run this PCA on training set only 
# Then applying same eigenvectors to the test set 

pca = PCA(n_components=100)
GMtrain_compressed = pca.fit_transform(GMtrain) # Fit the data for the training set 
GMtest_compressed = pca.transform(GMtest) # Fit the data for the test set

# Notes to self: 
# Fit_transform should take care of scaling. 
# Fit runs the PCA onto a particular dataset. Transform applies that dimensionality 
# reduction onto a different dataset.  

#Some checks to see if the number of components makes sense
#print("shape of gm_maps_masked is %f" % gm_maps_masked.shape)
#print("shape of GMtrain_compressed is %g" % GMtrain_compressed.shape)
#print("shape of GMtest_compressed is %r" % GMtest_compressed.shape)

#Plotting the Cumulative Summation of the Explained Variance
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance') #for each component
plt.title('PCA - Oasis Grey Matter Explained Variance')
#plt.savefig("C:Users/rwick/Documents/GitHubNew/rwickens-sMRI-PET/cumulative_sum.png")
plt.show()

#Create a pandas dataframe with the loadings for each component
df = pd.DataFrame(GMtrain_compressed)

#Creating a plot showing variance explained by first 20 components. 

shortened_components = pca.explained_variance_ratio_[:20] #will this get the first 20 items of the array...
mylist = []
for i in range(20): 
    mylist.append("PCA%i" % i)

plt.figure(figsize=(15,5))
df = pd.DataFrame({'var':shortened_components,
             'PC':mylist})
sns.barplot(x='PC',y="var", 
           data=df, color="c")
plt.show()
#plt.savefig("C:Users/rwick/Documents/GitHubNew/rwickens-sMRI-PET/barplot.png")

pipe = Pipeline([
    ('PCA', PCA(random_state=123)), 
    ('svr', svr)])

### Fit and predict
pipefit = pipe.fit(GMtrain, age_train)
age_pred_pipe = pipe.predict(GMtest) # returns a numpy array

# Measure accuracy with cross-validation
# This is to test the fit of the training data ONLY! 
cv_scores_train = cross_val_score(pipe, GMtrain, age_train) 
training_prediction_accuracy = np.mean(cv_scores_train)
print("The accuracy of the TRAINING svm model in predicting training data is %a" % training_prediction_accuracy)

# Running cross-validation on test data 
cv_scores_test = cross_val_score(pipe, GMtest, age_test)

# Return the corresponding mean prediction accuracy
prediction_accuracy = np.mean(cv_scores_test)
print("=== SVM ===")
print("Prediction accuracy: %f" % prediction_accuracy)
print("")

difference = (training_prediction_accuracy - prediction_accuracy)
print("The difference between testing and training accuracy is %c" % difference)

# Now, creating the inverse image to show the svr weights back on a horizontal slice of a scan

# retrieving the coefficients, or weights, of the linear svr 
coef = svr.coef_

# reverse feature selection
coef = pca.inverse_transform(coef)
print(coef) #just curious what this outputs

# reverse masking

nifti_masker = NiftiMasker(
    standardize=False,
    smoothing_fwhm=2,
    memory='nilearn_cache')  # cache options

fit_transform_nifti_masker = nifti_masker.fit_transform(gray_matter_map_filenames)

weight_img = fit_transform_nifti_masker.inverse_transform(coef) # type nibabel.nifti1.Nifti1Image

# *******Uncomment this out once you've saved oasis_dataset as an object onto disk?

# Create the figure based on the first subject's scan
bg_filename = bg_filename = gray_matter_map_filenames[0]
# bg_filename = "C:/Users/rwick/nilearn_data/oasis1/OAS1_0001_MR1/mwrc1OAS1_0001_MR1_mpr_anon_fslswapdim_bet.nii.gz"
z_slice = 0 # Horizontal slice of the brain
fig = plt.figure(figsize=(5.5, 7.5), facecolor='k')

# Hard setting vmax (upper bound for plotting) to highlight weights more 
display = plot_stat_map(weight_img, bg_img=bg_filename,
                        display_mode='z', cut_coords=[z_slice],
                        figure=fig, vmax=1)
display.title('SVM weights PCA SVR', y=1.2)
plt.savefig("C:Users/rwick/Documents/GitHubNew/rwickens-sMRI-PET/svm_inverse.png")

print("reached the end of the PCA-SVM program; if plots have not been made you should worry")

#------------------------------------------------

print("ANOVA + SVR")
# Define the prediction function to be used.
# Here we use a Support Vector Classification, with a linear kernel
svr2 = SVR(kernel="linear")

# Dimension reduction

# Remove features with too low between-subject variance
variance_threshold = VarianceThreshold(threshold=.01)

# Here we use a classical univariate feature selection based on F-test,
# namely Anova.
feature_selection = SelectKBest(f_regression, k=2000)

# We have our predictor (SVR), our feature selection (SelectKBest), and now,
# we can plug them together in a *pipeline* that performs the two operations
# successively:
from sklearn.pipeline import Pipeline
anova_svr = Pipeline([
            ('variance_threshold', variance_threshold),
            ('anova', feature_selection),
            ('svr', svr2)])

### Fit and predict
anova_svr.fit(gm_maps_masked, age)
age_pred_anovasvr = anova_svr.predict(gm_maps_masked)

coef_anovasvr = svr2.coef_
# reverse feature selection
coef_anovasvr = feature_selection.inverse_transform(coef_anovasvr)
# reverse variance threshold
coef_anovasvr = variance_threshold.inverse_transform(coef_anovasvr)
# reverse masking
# weight_img here xxx
weight_img_anovasvr = fit_transform_nifti_masker.inverse_transform(coef_anovasvr)

# Create the figure
bg_filename = gray_matter_map_filenames[0]
z_slice = 0

fig = plt.figure(figsize=(5.5, 7.5), facecolor='k')
# Hard setting vmax to highlight weights more
display = plot_stat_map(weight_img, bg_img=bg_filename,
                        display_mode='z', cut_coords=[z_slice],
                        figure=fig, vmax=1)
display.title("SVM weights ANOVA SVR", y=1.2)

# Measure accuracy with cross validation
cv_scores_anovasvr = cross_val_score(anova_svr, gm_maps_masked, age)

# Return the corresponding mean prediction accuracy
prediction_accuracy_anovasvr = np.mean(cv_scores_anovasvr)
print("=== ANOVA ===")
print("Prediction accuracy: %f" % prediction_accuracy_anovasvr)
print("")

### Inference with massively univariate model #################################
print("Massively univariate model")# Statistical inference
from nilearn.mass_univariate import permuted_ols
data = variance_threshold.fit_transform(gm_maps_masked)
neg_log_pvals, t_scores_original_data, _ = permuted_ols(
    age, data,  # + intercept as a covariate by default
    n_perm=2000,  # 1,000 in the interest of time; 10000 would be better
    n_jobs=1)  # can be changed to use more CPUs
signed_neg_log_pvals = neg_log_pvals * np.sign(t_scores_original_data)
signed_neg_log_pvals_unmasked = nifti_masker.inverse_transform(
    variance_threshold.inverse_transform(signed_neg_log_pvals))

# Show results
threshold = -np.log10(0.1)  # 10% corrected

fig = plt.figure(figsize=(5.5, 7.5), facecolor='k')

display = plot_stat_map(signed_neg_log_pvals_unmasked, bg_img=bg_filename,
                        threshold=threshold, cmap=plt.cm.RdBu_r,
                        display_mode='z', cut_coords=[z_slice],
                        figure=fig)
title = ("Negative $log_{10}$ p-values"
         "(Non-parametric + max-type correction)")
display.title(title, y=1.2)

n_detections = (signed_neg_log_pvals_unmasked.get_data() > threshold).sum()
print("%d detections" % n_detections)

show()

print("reached the end of the ANOVA-SVM program; if plots have not been made you should worry")
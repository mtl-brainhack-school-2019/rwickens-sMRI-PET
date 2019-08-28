#!/usr/bin/env python
# coding: utf-8

"""

This jupyter notebook includes code taken from the nilearn project, 
"Voxel-Based Morphometry on Oasis dataset", which predicts age from
grey matter morhpometry. Feature redux = k-best ANOVA, prediction function = SVM. 

I am tweaking the code to add the following:  
Train_test_split
Feature redux = PCA, prediction function = SVM.
Inverse image to show the voxels corresponding to support vector weights
Various plots

Eventually add k-fold validation
Try radial basis function support vector regressor? 
C and epsilon terms...

PCA appears to improve the model compared to the ANOVA for feature selection. 

To-do list: 
- Save oasis dataset as an object on computer; load  
- Why are your figures not showing up and not saving? 
- Difference between plt.show() versus just show()

"""

# Let's keep our notebook clean, so it's a little more readable!
import warnings
warnings.filterwarnings("ignore")

#All imports
from nilearn import datasets
from nilearn import plotting
from nilearn.plotting import plot_stat_map, show
import os
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
from sklearn.svm import SVR
from sklearn.decomposition import PCA
#from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

"""
#**********************************************************************
#*Only run this chunk of code once to download arrays to disk! 
oasis_dataset = datasets.fetch_oasis_vbm() #Selected all 403 subjects

#Let's plot the first subject's neuroimage for fun. 
subject1 = oasis_dataset.gray_matter_maps[0]
plotting.view_img(subject1)
#save this image somewhere

gray_matter_map_filenames = oasis_dataset.gray_matter_maps

nifti_masker = NiftiMasker(
    standardize=False,
    smoothing_fwhm=2,
    memory='nilearn_cache')  # cache options

gm_maps_masked_nondisk = nifti_masker.fit_transform(gray_matter_map_filenames)
age_nondisk = oasis_dataset.ext_vars['age'].astype(float)

#Save grey matter array onto disk
np.save("C:/Users/rwick/Documents/GitHub/rwickens-sMRI-PET/oasis.npy", gm_maps_masked_nondisk)

#Save age (our y-variable) array onto disk 
np.save("C:/Users/rwick/Documents/GitHub/rwickens-sMRI-PET/ages.npy", age_nondisk)

#Only run this chunk of code once to download arrays to disk! 
#*****************************************************
"""


#load grey matter array (x-variables) array from disk
gm_maps_masked = np.load("C:/Users/rwick/Documents/GitHub/rwickens-sMRI-PET/oasis.npy")
#Load age array (y-variable) from disk
age = np.load("C:/Users/rwick/Documents/GitHub/rwickens-sMRI-PET/ages.npy")

n_samples, n_features = gm_maps_masked.shape
print(age.shape)
print("%d samples, %d features" % (n_samples, n_features))

print(np.amax(gm_maps_masked))
#Hm. I wonder what units this is in. I was under the impression
#that tissue values represent probability, which should be between 0 and 1.

# Let's plot the distribution to see if this looks like a probability or not
plt.hist(gm_maps_masked, bins=10)
plt.savefig("C:Users/rwick/Documents/GitHubNew/rwickens-sMRI-PET/ournewdistributionfigure.png")
plt.show()

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
print("shape of gm_maps_masked is %f" % gm_maps_masked.shape)
print("shape of GMtrain_compressed is %g" % GMtrain_compressed.shape)
print("shape of GMtest_compressed is %r" % GMtest_compressed.shape)

#Plotting the Cumulative Summation of the Explained Variance
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance') #for each component
plt.title('PCA - Oasis Grey Matter Explained Variance')
plt.savefig("C:Users/rwick/Documents/GitHubNew/rwickens-sMRI-PET/cumulative_sum.png")
plt.show()

#Create a pandas dataframe with the loadings for each component
df = pd.DataFrame(GMtrain_compressed)
print(df)

#Creating a plot showing variance explained by first 20 components. 

print(len(pca.explained_variance_ratio_)) #should be 100, or whatever set to
print(type(pca.explained_variance_ratio_)) #should be numpy array
shortened_components = pca.explained_variance_ratio_[:20] #will this get the first 20 items of the array...

mylist = []
for i in range(20): 
    mylist.append("PCA%i" % i)
mylist

plt.figure(figsize=(15,5))
df = pd.DataFrame({'var':shortened_components,
             'PC':mylist})
sns.barplot(x='PC',y="var", 
           data=df, color="c")
plt.savefig("C:Users/rwick/Documents/GitHubNew/rwickens-sMRI-PET/barplot.png")

pipe = Pipeline([
    ('PCA', PCA(random_state=123)), 
    ('svr', svr)])                                                   
### Fit and predict
pipefit = pipe.fit(GMtrain, age_train)
print(type(pipefit)) #just curious what this would return as type  
age_pred_reb = pipe.predict(GMtest)
print(type(age_pred_reb)) #I'm guessing this will be an array of predicted y values

"""
if (len(age)==len(age_pred_reb)):
    #compute ordinary least squares here to get model accuracy
else:
    print("cannot compute residuals because of different array size!")
"""    

# Measure accuracy with cross-validation
# This is to test the fit of the training data ONLY! 
cv_scores_train = cross_val_score(pipe, GMtrain, age_train)
print("The accuracy of the training svm model in predicting training data is %a" % cv_scores_train)

# Measure accuracy with cross validation
# Running cross-validation on TEST DATA - the interesting part! 
cv_scores_test = cross_val_score(pipe, GMtest, age_test)
print("The accuracy of the training svm model in predicting test data is %d" % cv_scores_test)

difference = (cv_scores_train - cv_scores_test)

print("The difference between testing and training accuracy is %c" % difference)

# Return the corresponding mean prediction accuracy
prediction_accuracy = np.mean(cv_scores_test)
print("=== SVM ===")
print("Prediction accuracy: %f" % prediction_accuracy)
print("")

# retrieving the coefficients, or weights, of the linear svr 
coef = svr.coef_

# reverse feature selection
coef = pca.inverse_transform(coef)
print(coef) #just curious what this outputs

#reprinting this for sake of creating isolated code 
nifti_masker = NiftiMasker(
    standardize=False,
    smoothing_fwhm=2,
    memory='nilearn_cache')  # cache options

"""
*******Uncomment this out once you've saved oasis_dataset as an object onto disk
# reverse masking
weight_img = nifti_masker.inverse_transform(coef)
print(type(weight_img)) #Curious, I expect this to be a two-dimensional array (voxel & weight)

# Create the figure based on the first subject's scan
bg_filename = oasis_dataset.gray_matter_maps[0]
z_slice = 0 #Horizontal slice of the brain

fig = plt.figure(figsize=(5.5, 7.5), facecolor='k')

# Hard setting vmax to highlight weights more - in other words, normalizing?
display = plot_stat_map(weight_img, bg_img=bg_filename,
                        display_mode='z', cut_coords=[z_slice],
                        figure=fig, vmax=1)
display.title('SVM weights', y=1.2)
plt.savefig("C:Users/rwick/Documents/GitHubNew/rwickens-sMRI-PET/svm_inverse.png")
"""

print("reached the end of the PCA-SVM program; if plots have not been made you should worry")

#------------------------------------------------
"""
print("ANOVA + SVR")
# Define the prediction function to be used.
# Here we use a Support Vector Classification, with a linear kernel
from sklearn.svm import SVR
svr = SVR(kernel="linear")

# Dimension reduction
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression

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
            ('svr', svr)])

### Fit and predict
anova_svr.fit(gm_maps_masked, age)
age_pred = anova_svr.predict(gm_maps_masked)

coef = svr.coef_
# reverse feature selection
coef = feature_selection.inverse_transform(coef)
# reverse variance threshold
coef = variance_threshold.inverse_transform(coef)
# reverse masking
weight_img = nifti_masker.inverse_transform(coef)

# Create the figure
from nilearn.plotting import plot_stat_map, show
bg_filename = gray_matter_map_filenames[0]
z_slice = 0

fig = plt.figure(figsize=(5.5, 7.5), facecolor='k')
# Hard setting vmax to highlight weights more
display = plot_stat_map(weight_img, bg_img=bg_filename,
                        display_mode='z', cut_coords=[z_slice],
                        figure=fig, vmax=1)
display.title("SVM weights", y=1.2)

# Measure accuracy with cross validation
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(anova_svr, gm_maps_masked, age)

# Return the corresponding mean prediction accuracy
prediction_accuracy = np.mean(cv_scores)
print("=== ANOVA ===")
print("Prediction accuracy: %f" % prediction_accuracy)
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

#savefig here
show()
#should this be plot.show()
"""
# Structural MRI

My lab's focus is a PET radiotracer that shows great promise in quantifying brain acetylcholine.  

As of yet, my lab has used statistical parametric mapping as the main analysis. 

!['T-Maps'](http://www.ajnr.org/content/ajnr/early/2018/01/18/ajnr.A5527/F1.large.jpg)

I know there are more possibilities. I want to attempt to see what can be done to garner more attention to this radiotracer.   

At a birds-eye-view, my plans are two-fold:  

1. Grasp and replicate the current pipeline on my lab's past data.  
2. Learn about software and statistical techniques that I can implement in the lab.   

#Project 1: PREPROCESSING 

This part I'm less excited about. Why?

(1) I'm here because I like analyzing data, not because of pre-processing!
(2) Minctools isn't user-friendly
(3) The instructions I've gotten are rough  

<img src="https://github.com/mtl-brainhack-school-2019/rwickens-sMRI-PET/blob/master/Wtf.PNG?raw=true" width=1200> 

<img src="https://github.com/mtl-brainhack-school-2019/rwickens-sMRI-PET/blob/master/JackieChan.jpg"> width=40

Goals: 

- Understand my lab's preprocessing pipeline.   
- Successfully replicate it. 
  - To achieve this, I need to become comfortable working with minctools, CBrain, CIVET
- (If time allows, run data through ANTS, dartel)

!['Deliverable'](https://www.google.com/url?sa=i&source=images&cd=&ved=2ahUKEwiU_PXhpYbkAhUqheAKHRmPA8gQjRx6BAgBEAQ&url=https%3A%2F%2Fwww.visualpharm.com%2Ffree-icons%2Fdelivering&psig=AOvVaw1YnSDqz0qG-uF7DeyxGWcx&ust=1566007240834289))

###Deliverable 1:
A Jupyter notebook with clarified pipeline instructions. 
- Involves bash commands. 
- Briefly explains the rationale behind the processing steps. 

This will hopefully ease stress of future lab members. 

#PROJECT 2: STRUCTURAL IMAGE ML ANALYSIS. 

After preprocessing, I plan to switch gears to machine learning on large structural MRI datasets, specifically in Alzheimer's disease populations. 

- Download PREVENT-AD or OASIS datasets
 - Feature extraction: I will stick to measures of cortical thickness, brain volume, and other basic morphology. 
 - Put these features into matrices using numpy and pandas
- Dimensionality reduction using PCA
- Machine learning with Nilearn 
    - SVM? Random forest?
-Matplotlib plots along the way to visualize correlation matrices / PCA, etc. 
- Type of analysis to be determined. SVM? Random forest? 

!['Random forest'](https://www.fromthegenesis.com/wp-content/uploads/2018/06/Random-Forest.jpg)

!['Deliverable'](https://www.google.com/url?sa=i&source=images&cd=&ved=2ahUKEwiU_PXhpYbkAhUqheAKHRmPA8gQjRx6BAgBEAQ&url=https%3A%2F%2Fwww.visualpharm.com%2Ffree-icons%2Fdelivering&psig=AOvVaw1YnSDqz0qG-uF7DeyxGWcx&ust=1566007240834289)

###Deliverable 2: 

A Jupyter notebook walking through each of the above steps, with plots saved inline.  

---------------------------

Medium: Present these jupyter notebooks in a lab meeting and provide them on Github and lab Dropbox.  

Summary of tools I want to learn: 

MINC, CBRAIN, CIVET, FREESURFER, DARTEL, SCIKIT LEARN, NILEARN
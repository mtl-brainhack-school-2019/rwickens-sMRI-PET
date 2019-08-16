# Structural MRI

My lab's focus is a PET radiotracer that shows great promise in quantifying brain acetylcholine.  

As of yet, my lab has used statistical parametric mapping as the main analysis. 

<img src="http://www.ajnr.org/content/ajnr/early/2018/01/18/ajnr.A5527/F1.large.jpg" width=500> 

I know there are other possibilities out there. I want to attempt to see what can be done to garner more attention to this radiotracer.   

At a birds-eye-view, my plans are as follows:  

1. Grasp and replicate the current pipeline on my lab's past data.  
2. Learn about software and statistical techniques that I can implement in the lab.   

#Project 1: LEARN PREPROCESSING PIPELINE 

This part I'm less excited about. Why?

1. I'm here because I like analyzing data, less so preprocessing it. 
2. Minctools isn't user-friendly
3. The instructions I've been passed down are rough. 

<img src="https://github.com/mtl-brainhack-school-2019/rwickens-sMRI-PET/blob/master/Wtf.PNG?raw=true" width=1200> 

<img src="https://github.com/mtl-brainhack-school-2019/rwickens-sMRI-PET/blob/master/JackieChan.jpg" width=800> 

Goals: 

- Understand my lab's preprocessing pipeline.   
- Successfully replicate it. 
  - To achieve this, I need to become comfortable working with minctools, CBrain, CIVET
- (If time allows, run data through ANTS, dartel)

##Deliverable 1

<img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQfDhejxni_K2Gr-ItywgveUqpeeN-6LBfab93Svi6WyHVBMZ62" width=150> 

A Jupyter notebook with clarified pipeline instructions. 
- Involves bash commands. 
- Briefly explains the rationale behind the processing steps. 

This will hopefully ease stress of future lab members. 

#PROJECT 2: STRUCTURAL MRI ANALYSIS - ML. 

I plan to then switch gears to machine learning on large structural MRI datasets, specifically in Alzheimer's disease populations. Soon I hope to apply ML to my lab's data, which involves Alzheimer's and Parkinson's disease subjects. 

- Download PREVENT-AD and OASIS datasets
 - Feature extraction: I will stick to measures of cortical thickness, brain volume, and other basic morphology. 
 - Put these features into matrices using numpy and pandas
- Dimensionality reduction using PCA
- I would like my analysis to predict which group the scan comes from based on these features. 
- Machine learning with Nilearn 
    - SVM? Random forest?
- Learn how to perform cross-validation thanks to break-out session
- Matplotlib plots along the way to visualize correlation matrices / PCA, etc. 

<img src="https://www.fromthegenesis.com/wp-content/uploads/2018/06/Random-Forest.jpg" width=500> 

##Deliverable 2

<img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQfDhejxni_K2Gr-ItywgveUqpeeN-6LBfab93Svi6WyHVBMZ62" width=150> 

A Jupyter notebook walking through each of the above steps, with plots saved inline.  

---------------------------

Medium: Present these jupyter notebooks in a lab meeting and provide them on Github and lab Dropbox.  

Summary of tools I want to learn: 

MINC, CBRAIN, CIVET, BASH, SCIKIT LEARN, NILEARN
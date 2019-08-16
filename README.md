# STRUCTURAL MRI EXPLORATION

PhD project: cholinergic PET imaging of people with REM sleep behaviour disorder (prodromal stage of Parkinson's / Lewy body dementia). Part of an ongoing Montreal RBD cohort study that tracks disease progression across time. 

My lab's particular focus is a new PET radiotracer (FEOBV) that shows great promise in quantifying brain cholinergic systems.    

As of yet, my lab has used statistical parametric mapping as the main analysis. 

<img src="http://www.ajnr.org/content/ajnr/early/2018/01/18/ajnr.A5527/F1.large.jpg" width=500> 

I know there are other possibilities out there. I want to attempt to see what can be done to garner more attention to this radiotracer.   

At a birds-eye-view, I have two intentions:  

1. Grasp the current pipeline and replicate it on existing data.  
2. Learn about software and statistical techniques that I can implement in the lab and bring to the longitudinal study.    

-----------------------

## PROJECT 1: LEARN CURRENT PREPROCESSING PIPELINE 

This part I'm less excited about. Why?  

- I'm at BrainHack because I like analyzing data, less so preprocessing it. (But, I need to get over this hump.)
- Minctools isn't user-friendly
- The instructions I've been passed down are a little rough. 

<img src="https://github.com/mtl-brainhack-school-2019/rwickens-sMRI-PET/blob/master/Wtf.PNG?raw=true" width=850> 

<img src="https://github.com/mtl-brainhack-school-2019/rwickens-sMRI-PET/blob/master/JackieChan.jpg" width=350> 

### Goals

1. Understand my lab's preprocessing pipeline.   
2. Successfully replicate it. 
    - Become comfortable working with minctools, CBrain, CIVET
3. If time allows, run data through ANTS, dartel

### Deliverable 1

<img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQfDhejxni_K2Gr-ItywgveUqpeeN-6LBfab93Svi6WyHVBMZ62" width=200> 

- A Jupyter notebook with clarified pipeline instructions. 
    - Code will be in bash
    - Will contain more comments
    - Will briefly explain the rationale behind the processing steps, with particular attention to order.  

This will hopefully ease stress of future lab members. 

Someday, I will write a script that automates these steps, eliminating the need for the jupyter script I will create. 

-----------------------

## PROJECT 2: STRUCTURAL MRI ANALYSIS - ML CLASSIFIER. 

I plan to then switch gears to machine learning on large structural MRI datasets of individuals with Alzheimer's disease (AD). I would like to create a classifier to determine whether scan comes from an individual with AD or a healthy control. 

### Goals

- Download PREVENT-AD and OASIS datasets
 - Feature extraction
    - Basic morphology: cortical thickness, brain volume, etc. 
 - Put these features into workable matrices using numpy and pandas
- Dimensionality reduction using PCA
- Enter remaining features into model
    - Model type: SVM, random forest?  
- Learn about cross-validation techniques thanks to break-out session
- Nilearn to implement the above  
- Matplotlib plots along the way to visualize correlation matrices, model error, ROC curve, etc.

<img src="https://www.fromthegenesis.com/wp-content/uploads/2018/06/Random-Forest.jpg" width=500>

If extra time allows, delve into: 

- Longitudinal data 
- Model to predict onset of disease conversion
    - Survival analysis / trees?
    - Cox / hazards functions?

### Deliverable 2


<img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQfDhejxni_K2Gr-ItywgveUqpeeN-6LBfab93Svi6WyHVBMZ62" width=200> 

A Jupyter notebook walking through each of the above steps, with plots saved inline.

-------------

### Project medium


I will present these jupyter notebooks in a lab meeting and provide all documents on Github.

During my PhD, I hope to apply these ML techniques to my PET imaging, and to get involved in the cohort study.    


-------------------------------------

# COMMENTS? 
# FEEDBACK? 
# THANKS ! 
---------------------------------------------

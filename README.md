# STRUCTURAL MRI EXPLORATION

My lab's focus is a PET radiotracer that shows great promise in quantifying brain cholinergic systems. We are interested in applying this technique for imaging neurodegenerative conditions. 

As of yet, my lab has used statistical parametric mapping as the main analysis. 

<img src="http://www.ajnr.org/content/ajnr/early/2018/01/18/ajnr.A5527/F1.large.jpg" width=500> 

I know there are other possibilities out there. I want to attempt to see what can be done to garner more attention to this radiotracer.   

At a birds-eye-view, my plans are as follows:  

1. Grasp the current pipeline and replicate it on previous data.  
2. Learn about software and statistical techniques that I can implement in the lab.   

-----------------------

## PROJECT 1: LEARN PREPROCESSING PIPELINE 

This part I'm less excited about. 

1. I'm here because I like analyzing data, less so preprocessing it. (But, I need to get over this hump.)
2. Minctools isn't user-friendly
3. The instructions I've been passed down are rough. 

<img src="https://github.com/mtl-brainhack-school-2019/rwickens-sMRI-PET/blob/master/Wtf.PNG?raw=true" width=1000> 

<img src="https://github.com/mtl-brainhack-school-2019/rwickens-sMRI-PET/blob/master/JackieChan.jpg" width=400> 

### Goals

- Understand my lab's preprocessing pipeline.   
- Successfully replicate it. 
  - Become comfortable working with minctools, CBrain, CIVET
- (If time allows, run data through ANTS, dartel)

### Deliverable 1

<img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQfDhejxni_K2Gr-ItywgveUqpeeN-6LBfab93Svi6WyHVBMZ62" width=200> 

A Jupyter notebook with clarified pipeline instructions. 
- Code will be in bash
- Will contain more comments
- Will briefly explain the rationale behind the processing steps, with particular attention to order.  

This will hopefully ease stress of future lab members. 

In the future, I would like to write a script to automate our lab's preprocessing.

-----------------------

## PROJECT 2: STRUCTURAL MRI ANALYSIS - ML CLASSIFIER. 

I plan to then switch gears to machine learning on large structural MRI datasets of individuals with Alzheimer's disease (AD). I would like to create a classifier to determine whether scan comes from an individual with AD or a healthy control. Soon I hope to apply ML to my lab's data, which involves Alzheimer's and Parkinson's disease subjects. 

### Goals

- Download PREVENT-AD and OASIS datasets
 - Feature extraction
    - Basic morphology: cortical thickness, brain volume, etc. 
 - Put these features into workable matrices using numpy and pandas
- Dimensionality reduction using PCA
- Enter remaining features into model
    - Model type: SVM, random forest?  
- Learn about cross-validation techniques thanks to break-out session
- Nilearn to analyse the model and perform validation  
- Matplotlib plots along the way to visualize correlation matrices, model error, etc. 

<img src="https://www.fromthegenesis.com/wp-content/uploads/2018/06/Random-Forest.jpg" width=500> 

### Deliverable 2


<img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQfDhejxni_K2Gr-ItywgveUqpeeN-6LBfab93Svi6WyHVBMZ62" width=200> 

A Jupyter notebook walking through each of the above steps, with plots saved inline.  

-------------

### Project medium


I will present these jupyter notebooks in a lab meeting and provide them to the lab via Github.


-------------------------------------


# COMMENTS? 
# FEEDBACK? 
# THANKS ! 
---------------------------------------------

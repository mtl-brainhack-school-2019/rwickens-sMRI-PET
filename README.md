# Structural MRI

My lab's focus is a PET radiotracer that is promising in imaging the cholinergic system. 

As of yet, my lab has used statistical parametric mapping as the main analysis. 

!['T-Maps'](http://www.ajnr.org/content/ajnr/early/2018/01/18/ajnr.A5527/F1.large.jpg)

I know there are more possibilities. I want to attempt to see what can be done to garner more attention to this radiotracer.   

Abstractly, my plans have two phases: 

1. Grasp and replicate the current pipeline on my lab's past data.  
2. Learn about software and statistical techniques that I can implement in the lab.   

#Project 1: PREPROCESSING 

This part I'm less excited about. Why?

(1) I'm here because I like analyzing data, not because of pre-processing.
(2) Minctools isn't user-friendly
(3) The instructions I've gotten are rough  

<img src="https://github.com/mtl-brainhack-school-2019/rwickens-sMRI-PET/blob/master/Wtf.PNG?raw=true" width=1200> 

<img src="https://github.com/mtl-brainhack-school-2019/rwickens-sMRI-PET/blob/master/JackieChan.jpg"> width=60

Goals: 

- Understand my lab's preprocessing pipeline.   
- Successfully replicate it. 
  - To achieve this, I need to become comfortable working with minctools, CBrain, CIVET
- (If time allows, run data through ANTS, dartel)

!['Deliverable'](https://identixweb.com/wp-content/uploads/2019/02/order_delivery_date-copy-min.png)

###Deliverable 1:
A Jupyter notebook with clarified pipeline instructions. 
- Comprised of shell commands. 
- Briefly explains the rationale behind the processing steps. 

This will ease stress of future lab members and improve my own understanding. 

#PROJECT 2: STRUCTURAL IMAGE ML ANALYSIS. 

Machine learning on structural MRI data on large datasets of Alzheimer's disease scans.  

- Download PREVENT-AD or OASIS datasets
 - Feature extraction: I will stick to stu
 -  (using numpy and pandas)
- Dimensionality reduction using PCA
- Machine learning with Nilearn 
    - SVM? Random forest?
-Matplotlib plots used to visualize correlation matrices / PCA, etc. 
- Type of analysis to be determined. SVM? Random forest? 

!['Deliverable'](https://identixweb.com/wp-content/uploads/2019/02/order_delivery_date-copy-min.png)

###Deliverable 2: 

A Jupyter notebook walking through each of the above steps, with plots saved inline.  

---------------------------

Medium: Present these jupyter notebooks in a lab meeting and provide them on Github and lab Dropbox.  

Summary of tools I want to learn: 

MINC, CBRAIN, CIVET, FREESURFER, DARTEL, SCIKIT LEARN, NILEARN
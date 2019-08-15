# Structural MRI

- Which dataset do you want to analyze (if any)? 
  - Those of my lab (described below)
  - Then, the OASIS or PREVENT-AD datasets
-	Which tools do you want to learn? 
    - MINC, CIVET, CBRAIN, PYMINC, FREESURFER, DARTEL, NILEARN
- Which kind of deliverable do you want to implement: analysis, code, data, tutorial...? 
  - Project 1: Tutorial of jupyter code
  - Project 2: Well-commented jupyter file containing code and statistical analyses as well as plot outputs 2. 
- What kind of medium will you use to present the results? 
  - For both, I intend on giving a powerpoint presentation in which I run through these jupyter files. I plan on uploading the jupyter files on GitHub and disseminating it to my lab members.

#Project 1: sMRI / PET PREPROCESSING 

- Make sense of my lab's PET pipeline. Currently, the document is confusing and not user-friendly! 

Exhibit A: 

<img src="https://github.com/mtl-brainhack-school-2019/rwickens-sMRI-PET/blob/master/Wtf.PNG?raw=true" width=1200> 

<img src="https://github.com/mtl-brainhack-school-2019/rwickens-sMRI-PET/blob/master/JackieChan.jpg"> 

- Follow these steps and successfully pre-process my laboratory's sMRI and PET (FDG/NAV tracer) data
  - To achieve this, I need to become comfortable working with CBRAIN and CIVET
-	Run this same data thru Freesurfer and Dartel and/or ANTS
    - This will expose me to the more popular alternative of CIVET
- (If time allows, run data through MAGeT brain)

**Deliverable: a Jupyter notebook that improves the instruction sheet that I was given. Comprised of shell commands. Briefly explains the rationale behind the processing steps. This will help all future lab members of our lab!**

#PROJECT 2: 

- Download PREVENT-AD and OASIS datasets
- Feature extraction (use of python, numpy and pandas) 
- Dimensionality reduction using PCA
- Machine learning with Nilearn 
- Type of analysis to be determined

**Deliverable: a Jupyter notebook that shows each of the above steps.** 


---------------------------

Information about my lab's data: 

- Alzheimer’s disease (n = 12; 6 AD and 6 controls). 1.5T scanner. 
- Parkinson's disease (n = 18; 6 PD - cognitively normal, 6 PD - executive dysfunction, 6 controls). 3T scanner. 
- Idiopathic REM sleep behaviour disorder (n = 10; 5 with RBD and 5 controls). 3T scanner. 
 
I also have the following participant data:

- Demographic variables for all subjects (e.g., age, education)
- For the Alzheimer's study - NAV- and FDG-PET data
- For the REM sleep behaviour disorder: 
    - Clinical data: UPDRS scale, medical info, odour tests, hue tests, MoCA, and others
    - Movement-related data: Timed-up-and-go, purdue pegboard
    - Polysomnography data: sleep architecture, %EMG (phasic and tonic) during REM sleep
- For the Parkinson's study: UPDRS scale, autonomic function, Levodopa dose, disease duration, incomplete neuropsychological testing. 

-------------

My intention was to create a machine learning classifier that accurately classifies diagnosis (Alzheimer's, REM sleep behaviour disorder, Parkinson's, or healthy control) based on image features and clinical variables from my lab's dataset. However, Jake has informed me (2019-08-13) that this is not advisable due to sample size.   

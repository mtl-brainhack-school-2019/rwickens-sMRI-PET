# smri
My intention was to create a machine learning classifier that accurately classifies diagnosis (Alzheimer's, REM sleep behaviour disorder, Parkinson's, or healthy control) based on image features and clinical variables. However, Jake has informed me (2019-08-13) that this is not advisable due to my sample size.   

I have re-adjusted my goals as follows:  

Week 1: 

Pre-process my laboratory's sMRI and PET (FDG/NAV) data/
  Working in Linux to work with .mnc format/
  Follow the lab's current pipeline (includes CIVET) and internalize it/
Generate t-maps/

Week 2: 

Download PREVENT-AD and OASIS datasets
Pre-process MRI files to make them analysis-ready
Feature extraction
Dimensionality reduction
Machine learning with Nilearn 
Type of analysis to be determined

Information about my lab's data: 

Alzheimer’s disease (n = 12; 6 AD and 6 controls). 1.5T scanner. 
Parkinson's disease (n = 18; 6 PD - cognitively normal, 6 PD - executive dysfunction, 6 controls). 3T scanner. 
Idiopathic REM sleep behaviour disorder (n = 10; 5 with RBD and 5 controls). 3T scanner. 
 
I also have the following participant data:

Demographic variables for all subjects (e.g., age, education)
For the Alzheimer's study - NAV- and FDG-PET data
For the REM sleep behaviour disorder: 
    Clinical data : UPDRS scale, medical info, odour tests, hue tests, MoCA, and others
    Movement-related: Timed-up-and-go, purdue pegboard
    Polysomnography data: sleep architecture, %EMG (phasic and tonic) during REM sleep
For the Parkinson's study: UPDRS scale, autonomic function, Levodopa dose, disease duration, incomplete neuropsychological testing. 


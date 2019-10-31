#!/usr/bin/python

"""-----------INFORMATION FOR USER:----------- 
# Three inputs to run the program: weight (kg), dose (mCi), and patient folder (full path needed)
# Example input to run program: python /home/minc/projectfile/automate_PET.py 102 8.4 /home/minc/projectfolder/patientfolder
# Assumes you are in a project directory containing patients' folders.
# In this patient folder, you must have the IT file, TAL file, GRID file, and T1 file from CIVET,
# In the project directory, keep the json configuration file. In this file, you can change defaults (e.g, mask used, standard template used) 
# If the json config is not present, program will look for WM mask file (WM_0.99_new.mnc) nd MNI standard template file (mni_icbm152_t1_tal_nlin_sym_09c.mnc) in this project folder.
# Note: This program will overwrite files (if the file name of one of your ouputs already exists).   
"""

from pathlib import Path
from copy import deepcopy
import argparse
import subprocess
import os
from importlib import import_module
import json
import glob
import re

newest_minctools = max(glob.glob("/opt/minc/1.9.*"))
MINC_DIR = Path(newest_minctools)
DEFAULT_MINC_CONFIG = MINC_DIR / 'minc-toolkit-config.sh'
DEFAULT_MINC_BEST_LIN = MINC_DIR.joinpath('bin/bestlinreg_s')

parser = argparse.ArgumentParser('image processing inputs')
parser.add_argument('weight', type=float)
parser.add_argument('dose', type=float)
parser.add_argument('patient_folder')    

args = parser.parse_args()

def splice(path: Path, modifier) -> Path:
    dir_name = str(os.path.dirname(path))
    base_name = str(os.path.basename(path))
    base_count = base_name.count(".")
    if base_count > 1:
        base_count_until_last = base_count-1 
        newbase = base_name.replace('.','-', base_count_until_last)
        print("Your file names will be named ---", newbase, "--- extra periods '.' have been removed.")
        newbase = Path(newbase)
        dir_name = Path(dir_name)
        path = dir_name / newbase
    return path.parent.joinpath(path.stem + modifier).with_suffix(path.suffix)

def main(weight, dose, patient_folder):

    projectdir = Path(patient_folder).parent   
    print("projectdir is", projectdir)
    patient_dir = Path(patient_folder)
    print("patientdir is", patient_dir)

    json_path = projectdir / 'config.json'

    #if item in json file doesn't exist, depend on defaults... 
    if json_path.exists():
        config = json.load(json_path.open())
        PETsuffix = config['PET_SUFFIX']
        MRIsuffix = config['MRI_SUFFIX']
        talsuffix = config['TAL_SUFFIX']
        ITsuffix = config['IT_SUFFIX']
        MNItemplatepath = config['MNI_TEMPLATE_PATH']
        mask_or_atlas_path = config['MASK_OR_ATLAS_PATH']
        maskbinvalue = config['MASK_BIN_VALUE']
        mincconfigpath = config['MINC_CONFIG_PATH']
        mincbestlinregpath = config['MINC_BEST_LIN_REG_PATH']
        preferred_blur_list = config['PREFERRED_BLUR']
    else:
        print("json file not detected so going with defaults (MNI 152, WM_0.99 mask)")
        PETsuffix = "4D_MC01.mnc" 
        MRIsuffix = "_t1.mnc"   
        talsuffix = "t1_tal.xfm"
        ITsuffix = "_It.xfm"
        MNItemplatepath = projectdir / "mni_icbm152_t1_tal_nlin_sym_09c.mnc"
        mask_or_atlas_path = projectdir / "WM_0.99_new.mnc"
        maskbinvalue = 1
        mincconfigpath = DEFAULT_MINC_CONFIG
        mincbestlinregpath = DEFAULT_MINC_BEST_LIN
        preferred_blur_list = [4,6,8]
    
    PETpath = []
    MRIpath = []
    talpath = []
    ITpath = []
    gridpath = []

    PETpath = glob.glob(patient_folder + "/*" + PETsuffix)
    MRIpath = glob.glob(patient_folder + "/*" + MRIsuffix)
    talpath = glob.glob(patient_folder + "/*" + talsuffix)
    ITpath = glob.glob(patient_folder + "/*" + ITsuffix)
    gridpath = glob.glob(patient_folder + "/*It_grid_0.mnc")
    print(PETpath, MRIpath, talpath, ITpath, gridpath)

    if len(gridpath) == 0: 
        print("No grid file detected! Minc will likely raise an error about this during the transformations.")
    elif Path(gridpath[0]).parent != (Path(talpath[0]).parent or Path(ITpath[0]).parent) :
        print("Your grid file needs to be in the same folder as the TAL and IT files (CIVET)! This is likely to cause a problem later during the transformation stage")
    if len(PETpath) == 1:
        pass
    elif len(PETpath) > 1: 
        print("Multiple PET files ending in", PETsuffix, ". Check that there is only one patient's files in this patient folder.")
        raise SystemExit(0)
    else:
        print("No file found. Please check that your PETfile ends in", PETsuffix)
        raise SystemExit(0)
    if len(MRIpath) == 1:
        pass
    elif len(MRIpath) > 1: 
        print("Multiple PET files ending in", MRIsuffix, "check that the non-CIVET MRI file is not in this folder, and ensure that only one patient's files in this patient folder!")
        raise SystemExit(0)
    else:
        print("No file found. Please check that your PETfile ends in", MRIsuffix)
        raise SystemExit(0)
    if len(PETpath) == 1:
        pass
    elif len(talpath) > 1: 
        print("Multiple PET files ending in", talsuffix, "check that there is only one patient's files in this patient folder!")
        raise SystemExit(0)
    else:
        print("No file found. Please check that your PETfile ends in", talsuffix)
        raise SystemExit(0)
    if len(ITpath) == 1:
        pass
    elif len(ITpath) > 1: 
        print("Multiple PET files ending in", ITsuffix, "check that there is only one patient's files in this patient folder!")
        raise SystemExit(0)
    else:
        print("No file found. Please check that your PETfile ends in", ITsuffix)
        raise SystemExit(0)

    PETpath = Path(PETpath[0])
    MRIpath = Path(MRIpath[0])
    talpath = Path(talpath[0])
    ITpath= Path(ITpath[0])

    patient_code = str(os.path.basename(patient_folder)) 
    
    with open(patient_folder + "/" + patient_code + '_output_log.txt', 'w') as f:
        def bash_command(*args):
            bash_output = subprocess.check_output([str(c) for c in args], universal_newlines=True)
            bash_output = str(bash_output)
            print(bash_output)
            f.write(bash_output) 

        def bash_command_shell(*args):
            bash_output_shell = subprocess.check_output(args, shell=True, universal_newlines=True, executable='/bin/bash')
            bash_output_shell = str(bash_output_shell)
            print(bash_output_shell)
            f.write(bash_output_shell)
        
        mylist = []
        xfmlist = []
        mylist.append(PETpath)
        outputPETpath = mylist[-1]
        outputPETpath_xfm = outputPETpath.with_suffix('.xfm')
        xfmlist.append(outputPETpath_xfm)
        
        """
        #0. Split the PET file into single frames - un-comment if this step is desired
        
        number_frames = subprocess.check_output(['mincinfo', '-dimlength', 'time', PETpath], universal_newlines = True)
        number_frames = str(number_frames)
        print("number of frames is", number_frames)
        number_frames = re.sub("[^0-9.]", "", number_frames)
        print("the number of frames extracted is", number_frames)
        number_frames = int(number_frames)

        staticfiles = []
        for t in range(number_frames):
            t = str(t)
            staticfile = patient_folder+"/" + patient_code + "frame_{}.mnc".format(t)
            bash_command('mincreshape', '-clobber', '-dimrange', 'time=' + str(t), PETpath, staticfile)
        """
        """
        # 1. Change the file extension from .v to .mnc
        (Not needed unless dealing with .v files - such as with FDG)

        outputPETpath = PETpath.with_suffix('.mnc')
        mylist.append(outputPETpath)
        bash_command('ecattominc', '-short', PETpath, outputPETpath)
        """
        
        # 2. Average the PET frames

        outputPETpath = splice(outputPETpath, '_avg')
        mylist.append(outputPETpath)
        bash_command('mincaverage', '-clobber', mylist[-2], outputPETpath, '-avgdim', 'time')

        # 3. Take the SUV

        constant = dose * 1000 / weight
        print("dose * 1000 / weight = " + str(constant))
        outputPETpath = splice(outputPETpath, "_suv")
        mylist.append(outputPETpath)
        print("Go take a coffee break! The next couple of steps take 3-4 minutes to run.")
        bash_command('mincmath', '-clobber', '-div', mylist[-2], '-const', constant, outputPETpath)

        # 4. Automatic coregistration of subject PET to subject MRI to obtain .xfm files

        outputPETpath_xfm = splice(outputPETpath_xfm, "_autoreg")
        xfmlist.append(outputPETpath_xfm)
        outputPETpath = splice(outputPETpath, "_autoreg")
        mylist.append(outputPETpath)
        mincconfigpath = str(mincconfigpath)
        bash_command_shell("source " + str(mincconfigpath))
        bash_command(mincbestlinregpath, '-clobber', '-nmi', '-lsq6', mylist[-2], MRIpath, outputPETpath_xfm, outputPETpath)
        mylist_patient = deepcopy(mylist)  # Divergence in image processing (template space versus patient space)
        
        # 5. Linear and non-linear transformations to put the PET file into MNI space (toST = to Standard Template)
        
        outputPETpath_xfm = splice(outputPETpath_xfm, '_toST').with_suffix('.xfm')
        xfmlist.append(outputPETpath_xfm)
        bash_command('xfmconcat', '-clobber', xfmlist[-2], talpath, ITpath, outputPETpath_xfm)
        outputPETpath = splice(outputPETpath, "_ST")
        mylist.append(outputPETpath)
        bash_command('mincresample', '-clobber', mylist[-3], '-like', MNItemplatepath, '-transform', outputPETpath_xfm,
                     outputPETpath)

        # 6A. Take the SUVR in MNIspace        
        mask_SUV = subprocess.check_output(['mincstats', '-mask', str(mask_or_atlas_path), '-mask_binvalue', str(maskbinvalue), str(outputPETpath), '-mean'], universal_newlines = True)
        mask_SUV = str(mask_SUV)
        print("the output (mask_SUV) is", mask_SUV)
        #f.write("the output (mask_SUV) is", mask_SUV)
        mask_SUV = mask_SUV.strip()
        mask_SUV = re.sub("[^0-9.]", "", mask_SUV)
        print("the mask_SUV number extracted is", mask_SUV)
        #f.write("the mask_SUV number extracted is", mask_SUV)
        outputPETpath = splice(outputPETpath, '_SUVR')
        mylist.append(outputPETpath)
        bash_command('mincmath', '-clobber', '-div', '-const', mask_SUV, mylist[-2], outputPETpath)
            
        # 7A. Blur PET in select resolution(s) (default = 4, 6, and 8mm)
                
        templist = []
        deepcopyPETpath = deepcopy(outputPETpath) 
        for i in preferred_blur_list:
            blur_word = "_" + str(i)
            outputPETpath = splice(deepcopyPETpath, blur_word).with_suffix('')
            outputPETpath_for_list = splice(outputPETpath, '_blur').with_suffix('.mnc')
            templist.append(outputPETpath_for_list)
            bash_command('mincblur', '-clobber','-fwhm', i, mylist[-2], outputPETpath)
        for j in range(len(templist)):
            mylist.append(templist[j])
        
        # 8A. Finished. Show the patient's SUVR PET image on MNI template
        
        bash_command('register', mylist[-1], MNItemplatepath)

        # 6B - Take the SUVR in patient space
        
        PETsubjectmask = patient_folder + "/" + patient_code + "_subjectmask.mnc"
        bash_command('mincresample', '-clobber', '-like', mylist_patient[-2], '-nearest', '-transform', outputPETpath_xfm, '-invert_transformation', mask_or_atlas_path, PETsubjectmask)
        mask_SUV_patient = subprocess.check_output(['mincstats', '-mask', str(PETsubjectmask), '-mask_binvalue', str(maskbinvalue), str(mylist_patient[-2]), '-mean'], universal_newlines = True)
        mask_SUV_patient = str(mask_SUV_patient)
        print("the output (mask_SUV_patient) is", mask_SUV_patient)
        #f.write("the output (mask_SUV_patient) is", mask_SUV_patient)
        mask_SUV_patient = mask_SUV_patient.strip()
        mask_SUV_patient = re.sub("[^0-9.]", "", mask_SUV_patient)
        print("the mask_SUV_patient number extracted is", mask_SUV_patient)
        #f.write("the mask_SUV_patient number extracted is", mask_SUV_patient)         
        outputPETpath_patient = splice(mylist_patient[-1], '_patient_SUVR')
        mylist_patient.append(outputPETpath_patient)
        bash_command('mincmath', '-clobber', '-div', '-const', mask_SUV_patient, mylist_patient[-2], outputPETpath_patient)
        
        # 7B. Blur patient space in select resolution(s) (default = 4, 6, and 8mm)
        
        templist_patient = []
        deepcopyPETpath_patient = deepcopy(outputPETpath_patient)
        for i in preferred_blur_list: 
            blur_word_patient = "_" + str(i)           
            outputPETpath_patient = splice(deepcopyPETpath_patient, blur_word_patient).with_suffix('')
            outputPETpath_patient_for_list = splice(outputPETpath_patient, '_blur').with_suffix('.mnc')
            templist_patient.append(outputPETpath_patient_for_list)
            bash_command('mincblur', '-clobber', '-fwhm', i, mylist_patient[-2], outputPETpath_patient)
        for j in range(len(templist_patient)):
            mylist_patient.append(templist_patient[j])

        #8B. Show patient's SUVR PET image in their MRI space  
        
        bash_command('register', mylist_patient[-1], MRIpath)

main(**vars(args))

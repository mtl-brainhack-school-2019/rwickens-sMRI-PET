#!/usr/bin/python

# ^ won't be needed with _install_PET.py file 



"""-----------INFORMATION FOR USER: 

# PET file must have multiple frames. MRI file has already been run on CIVET

# You can adjust defaults on the configuration json file.

# Example bash input: python automate_PET_10-01.py 102.0 8.4 "ROYM"

# Weight and dose inputted are float (must have a decimal). (Check if this is even necessary)

# Blur will be done at 4, 6, and 8 mm. 

# Assumes you are in a directory containing patients' folders. Keep the py script, installer script, and json file in the larger directory.

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

parser.add_argument('folder')    



args = parser.parse_args()



# ISSUE 1 : getting weird outputs from os.path.basename/dirname; encoding issue? 

def splice(path: Path, modifier) -> Path:

    return path.parent.joinpath(path.stem + modifier).with_suffix(path.suffix)

    """

    #path = Path(path)

    dir_name = os.path.dirname

    base_name = os.path.basename

    if dir_name.count(".") > 1:

        print("Somewhere in your path, your folders have a '.' that should be removed")

    if base_name.count(".") > 1:

        base_name.replace(".","")

    path = base_name.append(dir_name)

    path = Path(path)

    """



def main(weight, dose, folder):



    workdir = Path(os.getcwd())

    print("workindir is")

    print(workdir)

    patient_dir = Path(os.getcwd()).joinpath(folder)

    print("patientdir is")

    print(patient_dir)

    if patient_dir.exists() == False:  

        print("problem with locating patient directory")

        raise SystemExit()



    json_path = Path(os.getcwd(), 'config.json')



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

    else:

        PETsuffix = "4D_MC01.mnc" 

        MRIsuffix = "_t1.mnc"   

        talsuffix = "t1_tal.xfm"

        ITsuffix = "_It.xfm"

        MNItemplatepath = workdir / "mni_icbm152_t1_tal_nlin_sym_09c.mnc"

        mask_or_atlas_path = workdir / "WM_0.99_new.mnc"

        maskbinvalue = 1

        mincconfigpath = DEFAULT_MINC_CONFIG

        mincbestlinregpath = DEFAULT_MINC_BEST_LIN

    

    PETpath = []

    MRIpath = []

    talpath = []

    ITpath = []



    for root, dirs, files in os.walk(patient_dir): 

        PETpath = glob.glob("**/*" + PETsuffix)

        MRIpath = glob.glob("**/*"+MRIsuffix)

        talpath = glob.glob("**/*"+talsuffix)

        ITpath = glob.glob("**/*"+ITsuffix)

    print(PETpath, MRIpath, talpath, ITpath)



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

    

    with open(folder + 'output.txt', 'w') as f:

        def bash_command(*args):

            bash_output = subprocess.check_output([str(c) for c in args], universal_newlines=True)

            #bash_output.decode("utf-8") #bc: error "write() argument must be str, not bytes"

            bash_output = str(bash_output)

            print(bash_output)

            f.write(bash_output) 

            # make sure check_output doesn't stop when warnings given



        def bash_command_shell(*args):

            bash_output_shell = subprocess.check_output([str(c) for c in args], shell=True, universal_newlines=True, executable='/bin/bash')

            print(bash_output_shell)

            f.write(bash_output_shell)

        

        mylist = []

        xfmlist = []

        mylist.append(PETpath)

        outputPETpath = mylist[-1]

        outputPETpath_xfm = outputPETpath.with_suffix('.xfm')

        xfmlist.append(outputPETpath_xfm)

        

        """

        (Not needed unless dealing with .v files - such as with FDG)

        # 1. Change the file extension from .v to .mnc

        outputPETpath = PETpath.with_suffix('.mnc')

        mylist.append(outputPETpath)

        bash_command('ecattominc', '-short', PETpath, outputPETpath)

        """

        

        # 2. Average the PET frames

        outputPETpath = splice(outputPETpath, '_avg')

        mylist.append(outputPETpath)

        bash_command('mincaverage', mylist[-2], outputPETpath, '-avgdim', 'time')



        # 3. Take the SUV

        constant = dose * 1000 / weight

        print("dose * 1000 / weight = " + str(constant))

        outputPETpath = splice(outputPETpath, "_suv")

        mylist.append(outputPETpath)

        print("Go take a coffee break! The next couple of steps take 3-4 minutes to run.")

        bash_command('mincmath', '-div', mylist[-2], '-const', constant, outputPETpath)



        # 4. Automatic coregistration of subject PET to subject MRI to obtain .xfm files

        outputPETpath_xfm = splice(outputPETpath_xfm, "_autoreg")

        xfmlist.append(outputPETpath_xfm)

        outputPETpath = splice(outputPETpath, "_autoreg")

        mylist.append(outputPETpath)

        bash_command_shell("source " + mincconfigpath)

        #bash_command('source', mincconfigpath)

        bash_command(mincbestlinregpath, '-nmi', '-lsq6', mylist[-2], MRIpath, outputPETpath_xfm, outputPETpath)

        mylist_patient = deepcopy(mylist)  # Divergence in image processing (template space versus patient space)



        # 5. Linear and non-linear transformations to put the PET file into MNI space (toST = to Standard Template)

        outputPETpath_xfm = splice(outputPETpath_xfm, '_toST').with_suffix('.xfm')

        xfmlist.append(outputPETpath_xfm)

        bash_command('xfmconcat', xfmlist[-2], talpath, ITpath, outputPETpath_xfm)

        outputPETpath = splice(outputPETpath, "_ST")

        mylist.append(outputPETpath)

        bash_command('mincresample', mylist[-3], '-like', MNItemplatepath, '-transform', outputPETpath_xfm,

                     outputPETpath)



        # 6A. Take the SUVR in MNIspace

        mask_or_atlas_path = str(mask_or_atlas_path)

        maskbinvalue = str(maskbinvalue)

        mask_SUV = subprocess.check_output(['mincstats', '-mask', mask_or_atlas_path, '-mask_binvalue', maskbinvalue, outputPETpath, '-mean'], universal_newlines = True)

        # write output to file

        # make multiple maskbinvalues possible here (csv); set shell=True and come up with method

        mask_SUV = str(mask_SUV)

        print("mask_SUV is", mask_SUV)

        mask_SUV = mask_SUV.strip()

        mask_SUV = re.sub("[^0-9]", "", mask_SUV)

        outputPETpath = splice(outputPETpath, '_SUVR')

        mylist.append(outputPETpath)

        bash_command('mincmath', '-div', '-const', mask_SUV, mylist[-2], outputPETpath)

            

        # 7A. Blur PET in 4, 6, and 8mm

                

        blurlist = [4, 6, 8]

        templist = []

        deepcopyPETpath = deepcopy(outputPETpath) 

        for i in blurlist:

            blur_word = "_" + str(i)

            outputPETpath = splice(deepcopyPETpath, blur_word).with_suffix('')

            outputPETpath_for_list = splice(outputPETpath, '_blur').with_suffix('.mnc')

            templist.append(outputPETpath_for_list)

            bash_command('mincblur', '-fwhm', i, mylist[-2], outputPETpath)

        for j in range(len(templist)):

            mylist.append(templist[j])

        

        # 8A. Finished. Show the patient's SUVR PET image on MNI template

        bash_command('register', mylist[-1], MNItemplatepath)



        # 6B - Take the SUVR in patient space

        bash_command('mincresample', '-like', mylist_patient[-1], '-nearest', '-transform', outputPETpath_xfm, '-invert_transformation', mask_or_atlas_path, 'PET_subjectmask.mnc')

        outputPETpath_patient = splice(mylist_patient[-1], '_patient_SUVR')

        mylist_patient.append(outputPETpath_patient)

        mylist_patient2 = str(mylist_patient[-2])

        PETsubjectmask = 'PET_subjectmask.mnc'

        maskbinvalue = str(maskbinvalue)

        mask_SUV_patient = subprocess.Popen(['mincstats', '-mask', PETsubjectmask, '–mask_binvalue', maskbinvalue, mylist_patient2, '–mean'])       

        mask_SUV_patient = str(mask_SUV_patient)

        mask_SUV_patient = mask_SUV_patient.strip()

        mask_SUV_patient = re.sub("[^0-9]", "", mask_SUV_patient)        

        bash_command('mincmath', '-div', '-const', mask_SUV_patient, mylist_patient[-2], outputPETpath_patient)

        

        # 7B. Blur PET in 4, 6, and 8 mm for patient space image

        blurlist_patient = [4, 6, 8]

        templist_patient = []

        deepcopyPETpath_patient = deepcopy(outputPETpath_patient)

        for i in blurlist_patient: 

            blur_word_patient = "_" + str(i)           

            outputPETpath_patient = splice(deepcopyPETpath_patient, blur_word_patient).with_suffix('')

            outputPETpath_patient_for_list = splice(outputPETpath_patient, '_blur').with_suffix('.mnc')

            templist_patient.append(outputPETpath_patient_for_list)

            bash_command('mincblur', '-fwhm', i, mylist_patient[-2], outputPETpath_patient)

        for j in range(len(templist_patient)):

            mylist_patient.append(templist_patient[j])



        #8B. Show patient's SUVR PET image in their MRI space  

        bash_command('register', mylist_patient[-1], MRIpath)



main(**vars(args))


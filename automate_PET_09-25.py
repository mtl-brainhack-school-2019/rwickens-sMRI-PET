#!/usr/local/bin/python3

# Specifications: PET file is the one with multiple (6) frames. MRI file has already been run on CIVET
# Weight, dose, and maskbinvalue are numeric (float). The rest are string.
# There are default values for masknumber, mincconfigpath, mincbestlinregpath, and blur. They are set to minc version 16

from copy import deepcopy
import argparse
import subprocess
import os

def main(weight, dose, PETpath, MRIpath, talpath, ITpath, MNItemplatepath, maskpath, masknumber, mincconfigpath, mincbestlinregpath, blur_mm):

    subject_code = os.path.basename(PETpath)
    subject_code = subject_code[:6]
    print(subject_code)
    with open('output' + subject_code + '.txt', 'w') as f: 
    #with open('output.txt', 'w') as f:
        
        weight = float(weight)
        dose = float(dose)
        PETpath = str(PETpath)
        MRIpath = str(MRIpath)
        talpath = str(talpath)
        ITpath= str(ITpath)
        MNItemplatepath = str(MNItemplatepath)
        maskpath= str(maskpath)
        masknumber= str(masknumber)
        mincconfigpath= str(mincconfigpath)
        mincbestlinregpath= str(mincbestlinregpath)
        blur_mm= int(blur_mm)
        
        def bash_command(cmd):
            subprocess.run(cmd, shell=False, stdout=f, check=True)

        mylist = []
       
        # 1. Change the file extension from .v to .mnc
        outputPETpath = PETpath[:-1] + "mnc"
        mylist.append(outputPETpath)
        bash_command(['ecattominc', '-short', PETpath, outputPETpath])
        #bash_command('ecattominc -short', PETpath, outputPETpath)

        #  2. Average the PET frames
        outputPETpath = outputPETpath[:-4] + "_avg" + outputPETpath[-4:]
        mylist.append(outputPETpath)
        bash_command(['mincaverage', mylist[-2], outputPETpath, '-avgdim', 'time'])
        # bash_command('mincaverage', mylist[-2], outputPETpath, '-avgdim time')
        # output should be PETpath_avg.mnc

        # 3. Take the SUV
        constant = dose*1000/weight
        print("dose * 1000 / weight = " + constant) # output via python instead of thru bash 
        outputPETpath = outputPETpath[:-4] + "_suv" + outputPETpath[-4:]
        mylist.append(outputPETpath)
        bash_command(['mincmath', '-div', mylist[-2], '-const', constant, outputPETpath])
        # bash_command('mincmath -div', mylist[-2], '-const', constant, outputPETpath)
        # output should be PETpath_avg_suv.mnc
        # make sure constant gets passed in as int  

        # 4. Automatic coregistration of subject PET to subject MRI to obtain .xfm files
        outputPETpath_xfm = outputPETpath[:-4] + "_autoreg.xfm"
        xfmlist = []
        xfmlist.append(outputPETpath_xfm)
        outputPETpath = outputPETpath[:-4] + "_autoreg" + outputPETpath[-4:]
        mylist.append(outputPETpath)
        bash_command(['source', mincconfigpath])
        # bash_command('source', mincconfigpath)  
        bash_command([mincbestlinregpath, '–nmi', '–lsq6', mylist[-2], MRIpath, outputPETpath_xfm, outputPETpath])
        # bash_command(mincbestlinregpath, '–nmi –lsq6', mylist[-2], MRIpath, outputPETpath_xfm, outputPETpath)
        # output should be PETpath_avg_suv_autoreg.mnc or .xfm
        
        #5. Linear and non-linear transformations to put the PET file into MNI space (toST = to Standard Template)
        outputPETpath_xfm = outputPETpath[-2]
        outputPETpath_xfm = outputPETpath_xfm[:-4] + "_toST.xfm"
        xfmlist.append(outputPETpath_xfm) 
        bash_command(['xfmconcat', xfmlist[-2], talpath, ITpath, outputPETpath_xfm])
        # bash_command('xfmconcat', xfmlist[-2], talpath, ITpath, outputPETpath_xfm)
        outputPETpath = outputPETpath[:-4] + "_ST" + outputPETpath[-4:]
        mylist.append(outputPETpath)
        bash_command(['mincresample', mylist[-2], '-like', MNItemplatepath, '-transform', outputPETpath_xfm, outputPETpath])
        # bash_command('mincresample', mylist[-2], '-like', MNItemplatepath, '-transform', outputPETpath_xfm, outputPETpath)
        # output should be PETpath_avg_suv_autoreg_toST.mnc or PETpath_avg_suv_autoreg_ST.mnc

        #6A. Take the SUVR in MNIspace
        mylist_patient = deepcopy(mylist)
        mask_SUV = subprocess.run(['mincstats', '-mask', maskpath, '-mask_binvalue', masknumber, outputPETpath, '-mean'] , capture_output=True, stdout=f, check=True)
        # mask_SUV = subprocess.run('mincstats -mask', maskpath, '-mask_binvalue', masknumber, outputPETpath, '-mean' , capture_output=True, shell=True, stdout=f, check=True)
        outputPETpath = outputPETpath[:-4] + "_SUVR" + outputPETpath[-4:]
        mylist.append(outputPETpath)
        bash_command(['mincmath', '-div', '-const', mask_SUV, mylist[-2], outputPETpath])
        # bash_command('mincmath -div -const', maskSUV, mylist[-2], outputPETpath)
        # make sure maskSUV gets processed as an int 
        # output should be PETpath_avg_suv_autoreg_ST_SUVR.mnc
       
        #6B - Take the SUVR in patient space
        PET_subjectmask = subject_code + "_subjectmask.mnc" # previously: 'PET_subjectmask.mnc'
        bash_command(['mincresample', '–like', mylist[-3], '–nearest', '–transform', outputPETpath_xfm, '–invert_transformation', maskpath, PET_subjectmask ]) 
        # bash_command('mincresample –like', mylist[-3], '–nearest –transform', outputPETpath_xfm, '–invert_transformation', maskpath, PET_subjectmask) 
        outputPETpath_patient = mylist_patient[-1]
        outputPETpath_patient = outputPETpath[:-4] + "_patient_SUVR" + outputPETpath[-4:]
        mylist_patient.append(outputPETpath_patient)
        mask_SUV_patient = subprocess.run(['mincstats', '–mask', PET_subjectmask, '–mask_binvalue', mylist[-3], mylist_patient[-2], '–mean'], capture_output=True, stdout=f, check=True)
        # mask_SUV_patient = subprocess.run(['mincstats –mask', PET_subjectmask, '–mask_binvalue', mylist[-3], mylist_patient[-2], '–mean', capture_output=True, shell=True, stdout=f, check=True)
        bash_command(['mincmath', '-div', '-const', mask_SUV_patient, mylist_patient[-2], outputPETpath_patient])
        # bash_command('mincmath -div -const', mask_SUV_patient, mylist_patient[-2], outputPETpath_patient)
        # make sure #mask_SUV_patient gets treated as int

        # 7A. Blur PET
        outputPETpath = outputPETpath[:-4] + "_blur" + str(blur_mm)
        bash_command(['mincblur', '-fwhm', blur_mm, mylist[-2], outputPETpath])
        # bash_command('mincblur -fwhm', blur_mm, mylist[-2], outputPETpath])

        # 8B. Blur PET for patient space image
        outputPETpath_patient = outputPETpath_patient[:-4] + "_blur" + str(blur_mm)    
        bash_command(['mincblur', '-fwhm', blur_mm, mylist_patient[-2], outputPETpath_patient])
        # bash_command('mincblur -fwhm', blur_mm, mylist[-2], outputPETpath)

        #9. Finished! Shows the subject's PET file on MNI template
        bash_command(['register', outputPETpath, MNItemplatepath])

parser = argparse.ArgumentParser('image processing inputs')
parser.add_argument('weight', type=float)
parser.add_argument('dose', type=float)
parser.add_argument('PETpath', type=str)
parser.add_argument('MRIpath', type=str)
parser.add_argument('talpath', type=str)
parser.add_argument('ITpath', type=str)
parser.add_argument('MNItemplatepath', type=str)
parser.add_argument('maskpath', type=str)
parser.add_argument('--masknumber', type=int, default = 1)
parser.add_argument('--mincconfigpath', type=str, default = "/opt/minc/1.9.16/minc-toolkit-config.sh")
parser.add_argument('--mincbestlinregpath', type=str, default = "/opt/minc/1.9.16/bin/bestlinreg_s")
parser.add_argument('--blur_mm', type=int, default=8)

args = parser.parse_args()

main(args.weight, args.dose, args.PETpath, args.MRIpath, args.talpath, args.ITpath, args.MNItemplatepath, args.maskpath, args.masknumber, args.mincconfigpath, args.mincbestlinregpath, args.blur_mm)

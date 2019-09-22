from copy import deepcopy
import argparse
import subprocess

# PET file is the one with 6 frames. MRI file is the one already run on CIVET
# Weight, dose, and maskbinvalue are numeric (float). The rest are string and must be surrounded by quotation marks.  
# Make sure that the grid file is in the CIVET folder. 

def main(weight, dose, PETpath, MRIpath, talpath, ITpath, MNItemplatepath, maskpath, masknumber, mincconfigpath, mincbestlinregpath):

    mylist = []
    
    # 1. Importer scan + tranformer en .mnc le FDG
    outputPETpath = PETpath[:-1] + "mnc"
    mylist.append(outputPETpath)
    ecattominc -short PETpath outputPETpath #remove .v, last character, and append "mnc" to change file extension
    # orig: ecattominc –short PETfilepath…3D.v …3D.mnc
    
    #  2. Average PET
    outputPETpath = outputPETpath[:-4] + "_avg" + outputPETpath[-4:]
    mylist.append(outputPETpath)
    mincaverage mylist[-2] outputPETpath –avgdim time
    # output should be PETpath_avg.mnc 
    # orig: mincaverage input.mnc output_av.mnc –avgdim time

    # 3. Do the SUV
    constant = dose*1000/weight
    constant | bc -l #output to user
    outputPETpath = outputPETpath[:-4] + "_suv" + outputPETpath[-4:]
    mylist.append(outputPETpath)
    mincmath –div mylist[-2] –const constant outputPETpath
    # output should be: PETpath_avg_suv.mnc

    """
    original
    echo dose*1000/weight | bc –l     
    (will give a constant)
    mincmath –div  –const constant ouput_suv.mnc
    """

    # 4. Automatic coregistration of subject PET to subject MRI to obtain .xfm files
    # Autoregister allows the PET to be adjusted to properly align with the MRI.
    # Paths to minc will have to be passed as arguments? Or script always run with certain files
    outputPETpath = outputPETpath[:-4] + "_autoreg" + outputPETpath[-4:]
    mylist.append(outputPETpath)
    outputPETpath_xfm = outputPETpath[:-4] + "_autoreg.xfm"
    xfmlist = []
    xfmlist.append(outputPETpathxfm)
    source mincconfigpath
    mincbestlinregpath –nmi –lsq6 mylist[-2] MRIpath outputPETpathxfm outputPETpath
    # output should be: PETpath_avg_suv_autoreg.mnc or .xfm
    """
    original
    source /opt/minc/1.9.51/minc-toolkit-config.sh 
    /opt/minc/bin/bestlinreg_s –nmi –lsq6 PETfile_av.mnc subjectMRI_t1_fromCBRAIN.mnc ouput_autoreg.xfm output_autoreg.mnc
    """

    #5. Transformation linear et non linear pour pouvoir mettre dans MNI space (toST = to Standard Template)
    outputPETpath_xfm = outputPETpath_xfm[:-4] + "_toST" + outputPETpath_xfm[-4:]
    xfmlist.append(outputPETpathxfm)
    xfmconcat xfmlist[-2] talpath ITpath outputPETpath_xfm
    # These are intermediate files 
    outputPETpath = outputPETpath[:-4] + "_ST" + outputPETpath[-4:]
    mylist.append(outputPETpath)
    mincresample mylist[-2] -like MNItemplatepath -transform outputPETpath_xfm outputPETpath

    # should be similar to xfmconcat PETpath_avg_suv_autoreg.xfm MNItemplatepath ITpath PETpath_avg_suv_autoreg_toST.xfm
    # mincresample PETpath_avg_suv_autoreg.mnc -like MNItemplatepath -transform PETpath_avg_suv_autoreg_toST.xfm PETpath_avg_suv_autoreg_ST.mnc

    """
    original
    xfmconcat PETfile.xfm ../../cbrain-output/subjectfolder/tranforms/linear/PETfile.xfm/subject_t1_tal.xfm cbrain-output/subjectfolder/tranforms/nonlinear/PETfile.xfm/subject_It.xfm PET_output_toST.xfm
    #This is an intermediate file 
    mincresample PETfile_av.mnc –like ./../mni_icbm152_tl_tal_nlin_sym_09c.mnc –transform PETfile_toST.xfm PET_ouput_ST.mnc
    #This causes the PET file to go onto the MNI template 
    #Minc resample is the application of the transformation matrix is applied to the PET image.
    """

    #6A. Faire le SUVR (in MNIspace)
    mylist_patient = deepcopy(mylist)
    maskSUVR = mincstats -mask maskpath -mask_binvalue masknumber outputPETpath -mean
    outputPETpath = outputPETpath[:-4] + "_SUVR" + outputPETpath[-4:]
    mylist.append(outputPETpath)
    mincmath -div -const maskSUVR mylist[-2] outputPETpath
    # Note: default maskbinvalue will be 1 unless otherwise specified 

    """
    # should look like PETpath_avg_suv_autoreg_ST.mnc PETpath_avg_suv_autoreg_ST_SUVR.mnc
    original
    mincstats –mask maskfile.mnc –mask_binvalue #ofthemask(or1) PETfile_ST.mnc –mean 
    mincmath –div –const meanvalueobtainbelow PETfile_ST.mnc PET_output_ST_SUVR.mnc
    """
    
    #6B - SUVR in patient space
    mincresample –like mylist_patient[-1] –nearest –transform xfmlist[-1] –invert_transformation maskpath PET_subjectmask.mnc 
    outputPETpath_patient = mylist_patient[-1]
    outputPETpath_patient = outputPETpath[:-4] + "_patient_SUVR" + outputPETpath[-4:]
    mylist_patient.append(outputPETpath_patient)
    mask_SUVR_patient = mincstats –mask PET_subjectmask.mnc –mask_binvalue mylist_patient[-2] –mean # replaced PETfile_av.mnc
    #will give mean value of radioactivity in the mask region, omit –mean to obtain all values)
    mincmath –div –const mask_SUVR_patient mylist_patient[-2] outputPETpath_path_patient

    """
    original
    mincresample –like PETfile_av.mnc –nearest –transform PETfile_toST.xfm –invert_transformation maskfile.mnc PET_ouput_subjectmask.mnc 
    mincstats –mask PET_subjectmask.mnc –mask_binvalue #ofthemask(or1) PETfile_av.mnc –mean 
    (will give mean value of radioactivity in the mask region, don’t put –mean to obtain all values)
    mincmath –div –const meanvalueobtainbelow PETfile_av.mnc PET_ouput_SUVR.mnc
    """

    # 8A. Blur PET
    outputPETpath = mylist[-1]
    outputPETpath = outputPETpath[:-4] + "_blur8" + outputPETpath[-4:]
    mincblur -fwhm 8 mylist[-1] outputPETpath
    # 8B. Blur PET for patient space image
    outputPETpath_patient = mylist_patient[-1]
    outputPETpath_patient = outputPETpath_patient[:-4] + "_blur8" + outputPETpath_patient[-4:]    
    mincblur -fwhm 8 mylist_patient[-1] output

    # original mincblur –fwhm 8 input.mnc ouput.mnc


parser = argparse.ArgumentParser('image processing inputs')
parser.add_argument('--weight', type=float)
parser.add_argument('--dose', type=float)
parser.add_argument('--PETpath', type=str)
parser.add_argument('--MRIpath', type=str)
parser.add_argument('--talpath', type=str)
parser.add_argument('--ITpath', type=str)
parser.add_argument('--MNItemplatepath', type=str)
parser.add_argument('--maskpath', type=str)
parser.add_argument('--masknumber', type=int, default = 1)
parser.add_argument('--mincconfigpath', type=str, default = "/opt/minc/1.9.17/minc-toolkit-config.sh")
parser.add_argument('--mincbestlinregpath', type=str, default = "/opt/minc/bin/bestlinreg_s")

args = parser.parse_args()

main(args.weight, args.dose, args.PETpath, args.MRIpath, args.talpath, args.ITpath, args.MNItemplatepath, args.maskpath, args.masknumber, args.mincconfigpath, args.mincbestlinreg)

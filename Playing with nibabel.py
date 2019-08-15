"""
Playing with Nibabel (python) on minc files.
Outputs stats that should be equivalent to mincstats command run on minc files.
"""

import numpy as np
import nibabel as nib
np.set_printoptions(precision=2, suppress=True)
img = nib.load('FEOBV_PD_00000201_t1.mnc')

data = img.get_data()
print(data.mean())
print(data.max())
print(data.min())

"""

Nibabel output: 

MEAN - 37.04384383788348
MAX - 447.0
MIN - 0.0

--------------------

#MINC output: 
Mask file:         (null)
Total voxels:      10813440
# voxels:          10813440
% of total:        100
Volume (mm3):      10813440
Min:               0
Max:               447
Sum:               400571382.7
Sum^2:             4.468324697e+10
Mean:              37.04384384
Variance:          2759.949284
Stddev:            52.53521947
CoM_voxel(z,y,x):  116.0619723 106.9590086 83.5769936
CoM_real(x,y,z):   9.735461372 46.3850648 -11.81005723

Histogram:         (null)
Total voxels:      10813440
# voxels:          10813440
% of total:        100
Median:            7.711159292
Majority:          5.02875
BiModalT:          70.06725
PctT [  0%]:       0.11175
Entropy :          5.986510924

"""

#myfile = nib.minc2.Minc2File('FEOBV_PD_00000201_t1.mnc')
#attempted to convert minc to nifti. But it convers to an object that isn't nifti.

#Checking out additional nibabel functions: 
header = img.header
print(header.get_data_shape())
print(img.shape)
print(header.get_data_dtype())
print(header.get_zooms())
data2d = data.reshape(np.prod(data.shape[:-1], data.shape[-1]))
print(header.get_data_shape())


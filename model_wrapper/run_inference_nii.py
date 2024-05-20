# Standard system imports
import csv
import glob
import json
import logging
import os

import numpy as np
import SimpleITK as sitk
import sys
import time
import torch
from scipy import io
# Standard routines for running CERR containers must be on PYTHONPATH
from scipy.io import loadmat

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import torch.nn.functional as F

# Local module with network weights, must be in cwd or on PYTHONPATH
from incre_MRRN import get_Incre_MRRN_Pytorch,normalize_data

# end module import statements  
num_args = len(sys.argv)
num_labels = 6

print(sys.argv)
if num_args == 1: # container
    softwarePath = '/software'
    dataPath = '/scratch'
    inputNiiPath = '/scratch/inputNii/'
    outputNiiPath = '/scratch/outputNii/'
    modelWeightsPath = '/software/model/seg_best_net_Seg_A.pth'
else:
    inputNiiPath = sys.argv[1]
    outputNiiPath = sys.argv[2]
    dataPath = os.path.join(inputNiiPath, os.pardir)
    scriptDir = os.path.dirname(os.path.abspath(__file__))
    wrapperDir = os.path.join(scriptDir, os.pardir)
    modelDir = os.path.join(wrapperDir, "model")
    modelWeightsPath = os.path.join(modelDir, 'seg_best_net_Seg_A.pth')
    print(modelWeightsPath)

algorithm_name = 'CT_LungOAR_incrMRRN'

## configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(algorithm_name)
labelthresh = 0.0
scanOffset = 1024

## Load network weights, get this from config
logger.info('Loading saved weights from ' + modelWeightsPath + '...')
model = get_Incre_MRRN_Pytorch() 
 
model.load_state_dict(torch.load(modelWeightsPath))
logger.info('Finish loading weight')


#load the data
niiGlob = glob.glob(os.path.join(inputNiiPath,'*.nii.gz'))
inputNiiFile = niiGlob[0]
inputImg = sitk.ReadImage(inputNiiFile)
scan_vol = sitk.GetArrayFromImage(inputImg) + scanOffset
vol_shape = scan_vol.shape
num_slices = vol_shape[0]
logger.debug('Input data shape: ' + str(vol_shape))
logger.info('Number of slices in loaded data: ' + str(num_slices))
logger.info('Input range: ' + str(np.min(scan_vol)) + ' ' + str(np.max(scan_vol)))

### min-max normalize
logger.info('Applying min-max normalization (-1,1)')

flipped_scan_vol = np.flip(np.flip(scan_vol,axis=1),axis=2)
flipped_scan_vol = np.moveaxis(flipped_scan_vol,[1,2],[2,1]) #Transpose image
norm_scan = normalize_data(flipped_scan_vol)
input_size = norm_scan.shape[1:]

logger.debug('norm_scan shape: ' + str(norm_scan.shape))


### Initialize array to hold model output in same shape as input original_scan_vol
labels_out = np.zeros((num_slices,input_size[0],input_size[1]))

### Begin processing
logger.info('Starting inference...')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

with torch.no_grad():
    for i in range(0,num_slices):

        logger.info('Slice number ' +  str(i + 1) + ' of ' + str(num_slices))

        # orient slice
        inputslice = np.ascontiguousarray(norm_scan[i].reshape(1,1,input_size[0],input_size[1]))

        # Convert numpy array to tensor, which used in pytorch (Jue)
        inputslice = torch.from_numpy(inputslice).cuda().float()

        #logger.debug('Single slice oriented, rotated 270 degrees, final shape: ' + str(inputslice.shape))

        label_array_probability = model(inputslice)
        label_array_probability = F.softmax(label_array_probability[0],dim=1)
        # convert tensor into numpy array
        label_array_probability = label_array_probability[0].cpu().numpy()
        #logger.debug('Size of returned label array struct: ' + str(label_array_probability.shape))

        #logger.debug('Thresholding label probability array by ' + str(labelthresh))
        labelslice = np.argmax(label_array_probability, axis = 0)
        #logger.debug('size of resulting labelslice ' + str(labelslice.shape))
        #labels_out[i] = np.rot90(labelslice.reshape(1,input_size[0],input_size[1]), k = 1, axes = (1,2))
        labels_out[i] = labelslice.reshape(1,input_size[0],input_size[1])

### Save output
maskOut = np.moveaxis(labels_out,[1,2],[2,1])
maskOut = np.flip(np.flip(maskOut,axis=1),axis=2)

inputFileName = os.path.basename(inputNiiFile)
outFilePrefix, ext = os.path.splitext(os.path.splitext(inputFileName)[0])
outFilePrefix = outFilePrefix.replace('scan', 'MASK')
outFile = os.path.join(outputNiiPath,outFilePrefix + '.nii.gz')
try:
        maskImg = sitk.GetImageFromArray(maskOut)	
        maskImg.CopyInformation(inputImg)
        sitk.WriteImage(maskImg, outFile)
except:
        logger.error('Unable to save Nii file output.')


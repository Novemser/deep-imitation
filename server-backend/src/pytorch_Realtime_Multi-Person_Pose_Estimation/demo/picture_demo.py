import os
import re
import sys
import cv2
import math
import time
import scipy
import argparse
import matplotlib
import numpy as np
import pylab as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from scipy.ndimage.filters import gaussian_filter

sys.path.append('/home/yaosy/Document/pytorch-EverybodyDanceNow/src/pytorch_Realtime_Multi-Person_Pose_Estimation/')
from network.rtpose_vgg import get_model
from network.post import decode_pose
from training.datasets.coco_data.preprocessing import (inception_preprocess,
                                              rtpose_preprocess,
                                              ssd_preprocess, vgg_preprocess)
from network import im_transform
from evaluate.coco_eval import get_multiplier, get_outputs, handle_paf_and_heat


weight_name = './network/weight/pose_model.pth'

model = get_model('vgg19')     
model.load_state_dict(torch.load(weight_name))
model = torch.nn.DataParallel(model).cuda()
model.float()
model.eval()



# test_image = './readme/ski.jpg'
test_image = '/home/yaosy/Document/pytorch-EverybodyDanceNow/data/target/images/img_0002.png'
timeReadStart = time.time()
oriImg = cv2.imread(test_image) # B,G,R order

timeReadEnd = time.time()
readTime = timeReadEnd - timeReadStart
print('read pic time {:.3f}s'.format(readTime % 60))

shape_dst = np.min(oriImg.shape[0:2])

# Get results of original image
multiplier = get_multiplier(oriImg)

timeMultiEnd = time.time()
print('multi time {:.3f}s'.format((timeMultiEnd - timeReadEnd) % 60))

with torch.no_grad():
    timeOriOutStart = time.time()
    orig_paf, orig_heat = get_outputs(
        multiplier, oriImg, model,  'rtpose')

    timeOriOutEnd = time.time()      
    print('Origin img output time {:.3f}s'.format((timeOriOutEnd - timeOriOutStart) % 60))
    # Get results of flipped image
    swapped_img = oriImg[:, ::-1, :]
    
    timeFlipOutStart = time.time()
    flipped_paf, flipped_heat = get_outputs(multiplier, swapped_img,
                                            model, 'rtpose')
    timeFlipOutEnd = time.time()
    print('Flip img output time {:.3f}s'.format((timeFlipOutEnd - timeFlipOutStart) % 60))

    # compute averaged heatmap and paf
    paf, heatmap = handle_paf_and_heat(
        orig_heat, flipped_heat, orig_paf, flipped_paf)
            
param = {'thre1': 0.1, 'thre2': 0.05, 'thre3': 0.5}
canvas, to_plot, candidate, subset = decode_pose(
    oriImg, param, heatmap, paf)
 
cv2.imwrite('result.png',to_plot)   


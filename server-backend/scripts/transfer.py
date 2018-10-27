# coding: utf-8

import os
import numpy as np
import torch
import time
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from collections import OrderedDict
from torch.autograd import Variable

os.chdir(os.path.dirname(os.path.realpath(__file__)))
pix2pixhd_dir = Path('../src/pix2pixHD/')

import sys
sys.path.append(str(pix2pixhd_dir))

from models.models import create_model
import util.util as util
from util.visualizer import Visualizer


with open('../data/test_opt.pkl', mode='rb') as f:
    opt = pickle.load(f)
opt.batchSize = 24

iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
visualizer = Visualizer(opt)
model = create_model(opt)

def model_inference(inputs):
    """
    Param
    inputs: a list contains labels e.g. [labels1, labels2, ..]
            p.s. labels don't have channel

    Return
    outputs: a list contains imgs e.g. [img1, img2, ...]
    """
    outputs = []
    num_input = len(inputs)
    data_label = torch.Tensor(inputs)
    data_label = data_label.unsqueeze(1)
    data_inst = torch.zeros(num_input)
    data = {'label': data_label, 'inst': data_inst}
    generated = model.inference(data['label'], data['inst'])
    for gen in generated:
        synthesized_image = util.tensor2im(gen)
        outputs.append(synthesized_image)
    return outputs

if __name__ == '__main__':
    import cv2
    test_file = '../data/syntest.png'
    inputs = [cv2.imread(test_file, 0)]
    outputs = model_inference(inputs)
    plt.imshow(outputs[0])
    plt.show()


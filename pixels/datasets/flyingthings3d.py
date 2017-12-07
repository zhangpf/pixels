import os
import glob
import collections
import random

from scipy.misc import imresize, imread
import numpy as np
from utils import *

MEAN_VALUE = np.float32(100.0)

class FlyingThings3D(object):
  def __init__(self, inputs, disp_dir, target_size,
         filter=False, disp_thres=300, cover_thres=0.25):

    self.get_samples(inputs, disp_dir)

    self.target_size = target_size
    self.target_height, self.target_width = target_size

    self.filter = filter
    self.disp_thres = disp_thres
    self.cover_thres = cover_thres
    self.type = (np.float32, np.float32, np.float32)
    self.shape = [ target_size + (3,), target_size + (3,), target_size + (1,) ]

  def get_samples(self, image_dir, disp_dir):
    left_names = sorted(glob.glob(image_dir + "/*/*/left/*"))
    right_names = sorted(glob.glob(image_dir + "/*/*/right/*"))
    disp_names = sorted(glob.glob(disp_dir + "/*/*/left/*"))
    assert len(left_names) == len(right_names)
    assert len(left_names) == len(disp_names)
    self.num_samples = len(left_names)
    self.samples = [left_names, right_names, disp_names]

  def __len__(self):
    return self.num_samples

  def __getitem__(self, key):
    return self.samples[key]

  def preprocess(self, inputs):
    left_name, right_name, disp_name = inputs
    left = imread(left_name, mode='RGB')
    right = imread(right_name, mode='RGB')
    disp = readPFM(disp_name)[0]
    left = imresize(left, self.target_size, interp='bilinear')
    right = imresize(right, self.target_size, interp='bilinear')
    disp = imresize(disp, self.target_size, interp='nearest')
    disp = disp[:,:, np.newaxis]
    left = (left - MEAN_VALUE) / 255
    right = (right - MEAN_VALUE) / 255
    disp = disp * self.target_width / np.float32(disp.shape[1])
    return (left, right, disp)


def train_and_val(data_dir, target_size):
  datasets = {}
  for prefix in ['TRAIN', 'TEST']:
    image_dir = os.path.join(data_dir, 'frames_cleanpass', prefix)
    disp_dir = os.path.join(data_dir, 'disparity', prefix)
    datasets[prefix] = FlyingThings3D(image_dir, disp_dir, target_size)
  
  return datasets['TRAIN'], datasets['TEST']
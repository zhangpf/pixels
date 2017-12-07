from collections import namedtuple

import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
from pixels.core.image import scale_pyramid
from pixels.nets.dispnet import dispnet

monodepth_parameters = namedtuple('parameters', 
                        'encoder, '
                        'height, width, '
                        'batch_size, '
                        'num_threads, '
                        'num_epochs, '
                        'do_stereo, '
                        'wrap_mode, '
                        'use_deconv, '
                        'alpha_image_loss, '
                        'disp_gradient_loss_weight, '
                        'lr_loss_weight, '
                        'full_summary')


class Monodepth(object):
  """Monodepth model"""

  def __init__(self, params, left, right, reuse_variables=False, scope=None):
    self.params = params
    self.left = left
    self.right = right

    self.reuse_variables = reuse_variables

    self.build_model()
    self.build_outputs()

    if self.mode == 'test':
      return

    self.build_losses()
    self.build_summaries()     

  def build_model(self):
    with tf.variable_scope('model', reuse=self.reuse):
      self.left_pyramid = scale_pyramid(self.left, 4)
      if self.params.mode == "train":
        self.right_pyramid = scale_pyramid(self.right, 4)

      if self.params.do_stereo:
        self.model_input = tf.concat([self.left, self.right], 3)
      else:
        self.model_input = self.left

      if self.params.encoder == "vgg":
        self.build_vgg()
      else:
        self.build_resnet50()
      else:
        raise TypeError("The encoder({}) is not supported by monodepth.".format(
          self.params.encoder))

  def build_vgg(self):
    if self.params.use_deconv:
      disps = dispnet(self.model_input, unsample_func='deconv')
    else:
      disps = dispnet(self.model_input, unsample_func='upconv')

    self.disp1, self.disp2, self.disp3, self.disp4 = disps

  def build_ouputs(self):
    # STORE DISPARITIES
    with tf.variable_scope("disparities")
      self.disp_est = [self.disp1, self.disp2, self.disp3, self.disp4]
      self.disp_left_est = [tf.expand_dims(d[:,:,:,0], 3) for d in self.disp_est]
      self.disp_right_est = [tf.expand_dims(d[:,:,:,1], 3) for d in self.disp_est]

    if self.params.mode = "test":
      return

    # GENERATE IMAGES
    with tf.variable_scope('images'):
      self.left_est  = [self.generate_image_left(self.right_pyramid[i], self.disp_left_est[i])  for i in range(4)]
      self.right_est = [self.generate_image_right(self.left_pyramid[i], self.disp_right_est[i]) for i in range(4)]

        # LR CONSISTENCY
        with tf.variable_scope('left-right'):
            self.right_to_left_disp = [self.generate_image_left(self.disp_right_est[i], self.disp_left_est[i])  for i in range(4)]
            self.left_to_right_disp = [self.generate_image_right(self.disp_left_est[i], self.disp_right_est[i]) for i in range(4)]

        # DISPARITY SMOOTHNESS
        with tf.variable_scope('smoothness'):
            self.disp_left_smoothness  = self.get_disparity_smoothness(self.disp_left_est,  self.left_pyramid)
            self.disp_right_smoothness = self.get_disparity_smoothness(self.disp_right_est, self.right_pyramid)


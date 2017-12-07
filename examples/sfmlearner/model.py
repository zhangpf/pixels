from __future__ import division

import tensorflow as tf

from pixels.nets.dispnet import dispnet
from pixels.depth.smooth import order_two_smoothness
from pixels.core.image import *
from pixels.core.pose import *
from pixels.depth.wrap import *
from tensorflow.contrib import slim

def pose_net(image1, image2, pose_scale=0.01):
  inputs = tf.concat([image1, image2], axis=3)
  with tf.name_scope('pose_net'):
    with slim.arg_scope([slim.conv2d],
                            normalizer_fn=None,
                            weights_regularizer=slim.l2_regularizer(0.05),
                            activation_fn=tf.nn.relu):
      conv1 = slim.conv2d(inputs, 16, 7, stride=2, scope='conv1')
      conv2 = slim.conv2d(conv1, 32, 5, stride=2, scope='conv2')
      conv3 = slim.conv2d(conv2, 64, 3, stride=2, scope='conv3')
      conv4 = slim.conv2d(conv3, 128, 3, stride=2, scope='conv4')
      conv5 = slim.conv2d(conv4, 256, 3, stride=2, scope='conv5')

      with tf.variable_scope('pose'):
        conv6 = slim.conv2d(conv5, 256, 3, stride=2, scope='conv6')
        conv7 = slim.conv2d(conv6, 256, 3, stride=2, scope='conv7')

        pose_pred = slim.conv2d(conv7, 6, 1, stride=1, scope='pose_pred', 
                    activation_fn=None)
        pose_final = pose_scale * tf.reduce_mean(pose_pred, [1, 2])

  return pose_final

class SFMLearner(object):
  def __init__(self, mode, images, intrinsic, num_scales=4, reuse=False, smooth_weight=0.5):
    self.mode = mode
    frames_length = images.get_shape().as_list()[3] // 3
    self.frames_length = frames_length
    
    self.images = tf.split(tf.cast(images, tf.float32), frames_length, axis=3)
    self.scaled_images = [ [ 
      scaled_resize_images(self.images[i], 1.0/(2**s)) for s in range(num_scales) ] 
          for i in range(frames_length) ]

    self.reuse = reuse
    self.smooth_weight = smooth_weight
    self.num_scales = num_scales
    self.intrinsic = intrinsic
    self.build_graph()

    if self.mode == 'train':
      self.build_loss()
      self.collect_summaries()

  def build_graph(self):
    
    if self.mode != 'test_depth':
      pose_net_reuse = self.reuse
      poses = []
      for i in range(1, self.frames_length):
        with tf.variable_scope("pose_prediction", reuse=pose_net_reuse): 
          pred_pose = pose_net(self.images[i], self.images[i-1])
          pose_net_reuse = True
          poses.append(pred_pose)

      poses = [ pose_vec2mat(p) for p in poses ]
      self.pred_poses = poses

    if self.mode != 'test_pose':
      disp_net_reuse = self.reuse
      depths = []
      for i in range(self.frames_length):
        with tf.variable_scope("depth_prediction", reuse=disp_net_reuse):
          pred_disp = dispnet(self.images[i])
          disp_net_reuse = True
          pred_depth = [ 1./d for d in pred_disp ]
          depths.append(pred_depth)
      self.pred_depths = depths


  def build_loss(self):
    smooth_loss = 0
    pixel_loss = 0
    
    for i in range(self.frames_length):
      smoothness = [ order_two_smoothness(self.pred_depths[i][j]) 
        for j in range(self.num_scales) ]
      smooth_loss += tf.add_n([ 
        smoothness[j] / (2 ** j) for j in range(self.num_scales) 
      ])

    for i in range(self.frames_length):
      curr_pose = self.pred_poses[i-1]
      for j in range(i-1,-1,-1):
        for s in range(self.num_scales):
          proj_image = projective_sampling(self.scaled_images[i][s], 
              self.pred_depths[i][s], curr_pose, scaled_resize_intrinsics(self.intrinsic, 1 / (2 ** s)))
          if s == 0 and j == 0:
            tf.summary.image('reconstruct_%d' % i, proj_image)
          proj_error = tf.abs(proj_image - self.scaled_images[j][s])

          pixel_loss += tf.reduce_mean(proj_error)

        if j != 0:
          curr_pose = tf.matmul(curr_pose, self.pred_poses[j-1])

    self.smooth_loss = smooth_loss
    self.pixel_loss = pixel_loss
    self.total_loss = smooth_loss * self.smooth_weight + pixel_loss

  def collect_summaries(self):
    tf.summary.scalar('total_loss', self.total_loss)
    tf.summary.scalar('smooth_loss', self.smooth_loss)
    tf.summary.scalar('pixel_loss', self.pixel_loss)

    tf.summary.image('disp', self.pred_depths[0][0])
    tf.summary.image('image', self.images[0])
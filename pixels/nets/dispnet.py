import tensorflow as tf
from tensorflow.contrib import slim

from pixels.core.image import *


def deconv(inputs, num_outputs, kernel_size, scope=None):
  with tf.variable_scope(scope, "deconv"):
    conv_trans = slim.conv2d_transpose(inputs, 
        num_outputs, kernel_size, scope="conv_trans")

  return conv_trans

def encode_block(inputs, num_outputs, kernel_size, scope=None):
  with tf.variable_scope(scope, "encode_block"):
    conv  = slim.conv2d(inputs,
        num_outputs, kernel_size, stride=1, scope="conv")
    convb = slim.conv2d(conv,
        num_outputs, kernel_size, stride=2, scope="convb")

  return convb

def decode_block(inputs, extras, num_outputs, kernel_size, scope=None):
  with tf.variable_scope(scope, "decode_block"):
    upconv = slim.conv2d_transpose(inputs, 
        num_outputs, kernel_size, stride=2, scope="upconv")
    
    if extras is None:
      concat = upconv
    elif isinstance(extras, (list, tuple)):
      concat = tf.concat([upconv] + list(extras), axis=3)
    elif isinstance(extras, tf.Tensor):
      concat = tf.concat([upconv, extras], axis=3)
    else:
      raise TypeError("The type of extras is not supported.") 

    iconv = slim.conv2d(concat, 
        num_outputs, kernel_size, stride=1, scope="iconv")

  return iconv

def disp_net_vgg(inputs, unsample_func='deconv', disp_scaling=10, min_disp=0.01):
  if callable(unsample_func):
    upsample = unsample_func
  elif unsample_func == 'deconv':
    upsample = deconv
  elif unsample_func == 'upconv':
    upsample = upconv
  else:
    raise ValueError("the upsample function is not supported.") 
  
  num_outputs = inputs.get_shape().as_list()[3] // 3

  conv1 = encode_block(inputs, 32,  7, scope='conv1')
  conv2 = encode_block(conv1,  64,  5, scope='conv2')
  conv3 = encode_block(conv2,  128, 3, scope='conv3')
  conv4 = encode_block(conv3,  256, 3, scope='conv4')
  conv5 = encode_block(conv4,  512, 3, scope='conv5')
  conv6 = encode_block(conv5,  512, 3, scope='conv6')
  conv7 = encode_block(conv6,  512, 3, scope='conv7')
  
  iconv7 = decode_block(conv7, conv6, 512, 3, scope='upconv7')
  iconv6 = decode_block(iconv7, conv5, 512, 3, scope='upconv6')
  iconv5 = decode_block(iconv6, conv4, 256, 3, scope='upconv5')
  iconv4 = decode_block(iconv5, conv3, 128, 3, scope='upconv4')
  disp4 = disp_scaling * slim.conv2d(iconv4, num_outputs, 3, stride=1, scope='disp4', activation_fn=tf.nn.sigmoid) + min_disp
  disp4_up = scaled_resize_images(disp4, 2, method='nn')
  iconv3 = decode_block(iconv4, [conv2, disp4_up], 64, 3, scope='upconv3')
  disp3 = disp_scaling * slim.conv2d(iconv3, num_outputs, 3, stride=1, scope='disp3', activation_fn=tf.nn.sigmoid) + min_disp
  disp3_up = scaled_resize_images(disp3, 2, method='nn')
  iconv2 = decode_block(iconv3, [conv1, disp3_up], 32, 3, scope='upconv2')
  disp2 = disp_scaling * slim.conv2d(iconv2, num_outputs, 3, stride=1, scope='disp2', activation_fn=tf.nn.sigmoid) + min_disp
  disp2_up = scaled_resize_images(disp2, 2, method='nn')
  iconv1 = decode_block(iconv2, disp2_up, 16, 3, scope='upconv1')
  disp1 = disp_scaling * slim.conv2d(iconv1, num_outputs, 3, stride=1, scope='disp1', activation_fn=tf.nn.sigmoid) + min_disp

  return [disp1, disp2, disp3, disp4]

def dispnet(inputs, encoder_type='vgg', disp_scaling=10, min_disp=0.01, 
            unsample_func='deconv'):
  with tf.name_scope('disp_net'):
    with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], 
        normalizer_fn=None, weights_regularizer=slim.l2_regularizer(0.05), 
        activation_fn=tf.nn.relu):
      if encoder_type == 'vgg':
        disp = disp_net_vgg(inputs, unsample_func=unsample_func,
            disp_scaling=disp_scaling, min_disp=min_disp)

  
  #disp = [ tf.maximum(d * disp_scaling, min_disp) for d in disp ]
  return disp

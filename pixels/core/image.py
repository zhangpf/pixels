import tensorflow as tf
import numpy as np

def image_pixel_coords(batch, height, width):
  x_t = tf.matmul(tf.ones(shape=tf.stack([height, 1])),
      tf.transpose(tf.expand_dims(tf.linspace(-1.0, 1.0, width), 1), [1, 0]))
  y_t = tf.matmul(tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
      tf.ones(shape=tf.stack([1, width])))
  x_t = (x_t + 1.0) * 0.5 * tf.cast(width - 1, tf.float32)
  y_t = (y_t + 1.0) * 0.5 * tf.cast(height - 1, tf.float32)
  coords = tf.stack([x_t, y_t], axis=0)
  return tf.tile(tf.expand_dims(coords, 0), [batch, 1, 1, 1])


def _resize_method(method):
  if isinstance(method, str):
    method = method.lower()
  if method == 'nn' or method == 'nearest_neighbor':
    return tf.image.ResizeMethod.NEAREST_NEIGHBOR
  elif method == 'bilinear' or method == 'bl':
    return tf.image.ResizeMethod.BILINEAR
  elif method == 'bc' or method == 'bicubic':
    return tf.image.ResizeMethod.BICUBIC
  elif method == 'a' or method == 'area':
    return tf.image.ResizeMethod.AREA
  else:
    raise ValueError("the method %s is not supported." % str(method))

def scaled_resize_images(images, ratio, method='a'):
  if isinstance(ratio, (tuple, list)):
    ratio_height, ratio_width = ratio
  else:
    ratio_height = ratio
    ratio_width = ratio 
  _, height, width, _ = images.get_shape().as_list()
  return tf.image.resize_images(images, 
      [ int(height*ratio_height), int(width*ratio_width) ],
      method=_resize_method(method))

def scale_pyramid(images, num_scales, ratio, method='a'):
  scale_images = [images]
  _, height, width, _ = images.get_shape().as_list()
  for i in range(num_scales - 1):
    scale_ratio = 1.0 / (ratio ** (i + 1))
    scale_images.append(scaled_resize(images, scale_ratio, method=method))

  return scale_images


def scaled_resize_intrinsics(intrinsics, ratio):
  if isinstance(ratio, (tuple, list)):
    ratio_height, ratio_width = ratio
  else:
    ratio_height = ratio
    ratio_width = ratio

  if isinstance(intrinsics, tf.Tensor):
    return intrinsics * tf.constant([[ratio_height], [ratio_width], [1.0]])
  elif isinstance(intrinsics, np.ndarray):
    return intrinsics * np.array([[ratio_height], [ratio_width], [1.0]])
  else:
    raise TypeError("The type of intrinsics is not supported.")
import tensorflow as tf

from pixels.core.image import *
from pixels.depth.sampling import bilinear_sampling_2d
from pixels.models.pinhole import pixel2camera, camera2pixel
from pixels.utils.utils import *

def projective_sampling(image, depth, pose, intrinsic):
	batch, height, width, _ = image.get_shape().as_list()

	pixel_coords = image_pixel_coords(batch, height, width)
	camera_coords = pixel2camera(pixel_coords, depth, intrinsic)
	camera_coords = homogeneous(camera_coords)

	camera_coords = tf.reshape(camera_coords, [batch, 4, -1])
	target_camera_coords = tf.matmul(pose, camera_coords)
	target_camera_coords = tf.reshape(target_camera_coords, [batch, 4, height, width])
	target_pixel_coords = camera2pixel(target_camera_coords, intrinsic)

	target_image = bilinear_sampling_2d(image, target_pixel_coords)
	return target_image

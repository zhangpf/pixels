
import tensorflow as tf

from pixels.utils.utils import *
import pixels

def pixel2camera(pixel_coords, depth, intrinsic):

	shape = pixel_coords.get_shape().as_list()

	if len(shape) == 3:
		batch, dim, _ = shape
	elif len(shape) == 4:
		batch, dim, _, _ = shape
		pixel_coords = tf.reshape(pixel_coords, [batch, dim, -1])
	else:
		raise ValueError("The rank of pixel coordinates tensor must be 3 or 4.") 

	if not dim in (2, 3):
		raise ValueError("The size of pixel coordinate must be 2 or 3 (for homogeneous).") 

	if dim == 2:
		pixel_coords = homogeneous(pixel_coords)

	depth = tf.reshape(depth, [batch, 1, -1])
	camera_coords = tf.matmul(tf.matrix_inverse(intrinsic), 
							  pixel_coords) * depth
	if len(shape) == 4:
		camera_coords = tf.reshape(camera_coords, [batch, -1, shape[2], shape[3]])
	return camera_coords


def camera2pixel(camera_coords, intrinsic):
	shape = camera_coords.get_shape().as_list()

	if len(shape) == 3:
		batch, dim, _  = shape
	elif len(shape) == 4:
		batch, dim, _, _ = shape
		camera_coords = tf.reshape(camera_coords, [batch, dim, -1])
	else:
		raise ValueError("The rank of pixel coordinates tensor must be 3 or 4.") 

	if not dim in (3, 4):
		raise ValueError("The size of pixel coordinate must be 2 or 3 (for homogeneous).") 

	if dim == 4:
		camera_coords = inhomogeneous(camera_coords)

	unnormalized_pixel_coords = tf.matmul(intrinsic, camera_coords)

	x_u = tf.slice(unnormalized_pixel_coords, [0, 0, 0], [-1, 1, -1])
	y_u = tf.slice(unnormalized_pixel_coords, [0, 1, 0], [-1, 1, -1])
	z_u = tf.slice(unnormalized_pixel_coords, [0, 2, 0], [-1, 1, -1])

	x_u = x_u / (z_u + 1e-10)
	y_u = y_u / (z_u + 1e-10)
	pixel_coords = tf.concat([x_u, y_u], axis=1)
	if len(shape) == 4:
		pixel_coords = tf.reshape(pixel_coords, [batch, 2, shape[2], shape[3]])
	return pixel_coords


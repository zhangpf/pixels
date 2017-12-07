
import tensorflow as tf


def gradient_x(img):
	return img[:,:,:-1,:] - img[:,:,1:,:]

def gradient_y(img):
	return img[:,:-1,:,:] - img[:,1:,:,:]

def neg_exp_weighted_smoothness(disp, image):
	gx = gradient_x(disp)
	gy = gradient_y(disp)
	image_gx = gradient_x(image)
	image_gy = gradient_y(image)

	weight_x = tf.exp(-tf.reduce_mean(tf.abs(image_gx), 3, 3, keep_dims=True))
	weight_y = tf.exp(-tf.reduce_mean(tf.abs(image_gy), 3, 3, keep_dims=True))

	return tf.reduce_mean(tf.abs(weight_y * gy)) + \
		   tf.reduce_mean(tf.abs(weight_x * gx))

def order_two_smoothness(disp):
	dx = gradient_x(disp)
	dy = gradient_y(disp)
	dx2 = gradient_x(dx)
	dxdy = gradient_y(dx)
	dydx = gradient_x(dy)
	dy2 = gradient_y(dy)

	return tf.reduce_mean(tf.abs(dx2)) + \
		   tf.reduce_mean(tf.abs(dxdy)) + \
		   tf.reduce_mean(tf.abs(dydx)) + \
		   tf.reduce_mean(tf.abs(dy2))

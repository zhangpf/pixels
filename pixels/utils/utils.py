import re

import numpy as np
import tensorflow as tf

def readPFM(file):
	file = open(file, 'rb')

	color = None
	width = None
	height = None
	scale = None
	endian = None

	header = file.readline().rstrip()
	if header == 'PF':
		color = True
	elif header == 'Pf':
		color = False
	else:
		raise Exception('Not a PFM file.')

	dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline())
	if dim_match:
		width, height = map(int, dim_match.groups())
	else:
		raise Exception('Malformed PFM header.')

	scale = float(file.readline().rstrip())
	if scale < 0: # little-endian
		endian = '<'
		scale = -scale
	else:
		endian = '>' # big-endian

	data = np.fromfile(file, endian + 'f')
	shape = (height, width, 3) if color else (height, width)

	data = np.reshape(data, shape)
	data = np.flipud(data)
	file.close
	return data, scale


def homogeneous(coords):
	shape = coords.get_shape().as_list()
	#TODO: parameter rank check
	assert len(shape) in (3, 4)
	ones = tf.ones([shape[0], 1] + shape[2:])
	return tf.concat([coords, ones], axis=1)

def inhomogeneous(coords):
	shape = coords.get_shape().as_list()
	#TODO: parameter rank check
	assert len(shape) in (3, 4)
	return tf.split(coords, [shape[1] - 1, 1], axis=1)[0]



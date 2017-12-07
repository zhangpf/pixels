import tensorflow as tf

def average_gradients(tower_grads):

	average_grads = []
	for grad_and_vars in zip(*tower_grads):
	# Note that each grad_and_vars looks like the following:
	#   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
		grads = []
		for g, _ in grad_and_vars:
			# Add 0 dimension to the gradients to represent the tower.
			expanded_g = tf.expand_dims(g, 0)

			# Append on a 'tower' dimension which we will average over below.
			grads.append(expanded_g)

		# Average over the 'tower' dimension.
		grad = tf.concat(axis=0, values=grads)
		grad = tf.reduce_mean(grad, 0)

		# Keep in mind that the Variables are redundant because they are shared
		# across towers. So .. we will just return the first tower's pointer to
		# the Variable.
		v = grad_and_vars[0][1]
		grad_and_var = (grad, v)
		average_grads.append(grad_and_var)
	return average_grads

def distribute_ops(num_gpus, tower_loss_func, *extra_func_args):
	extra_func_res = [[]] * len(extra_func_args)
	with tf.variable_scope(tf.get_variable_scope()) as sc:
		for i in xrange(num_gpus):
			with tf.device('/gpu:%d' % i):
				with tf.name_scope('%s_%d' % (TOWER_NAME, i)) as scope:

					loss = tower_loss_func(scope)
					grads = opt.compute_gradients(loss,)
					tower_grads.append(grads)

	grads = average_gradients(tower_grads)

	return grads, extra_func_res
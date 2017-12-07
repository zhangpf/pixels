import tensorflow as tf

def leaky_relu(x, alpha, name):
	return tf.maximum(alpha * x, x, name)

def L1_loss(x, y):
	return tf.reduce_mean(tf.abs(x - y))

# def initialize_uninitialized(sess):
# 	global_vars = tf.global_variables()
# 	is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
# 	not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]
	
# 	print [str(i.name) for i in not_initialized_vars] # only for testing
# 	if len(not_initialized_vars):
# 		sess.run(tf.variables_initializer(not_initialized_vars))


def initialize_uninitialized(sess):
	uninit_varnames = sess.run(tf.report_uninitialized_variables())
	
	if len(uninit_varnames):
		uninit_vars = [
			v for v in tf.global_variables() if v.name.split(':')[0] in uninit_varnames
		]
		sess.run(tf.variables_initializer(uninit_vars))
import tensorflow as tf

n_inputs = 3
n_neurons = 5

x0 = tf.placeholder(tf.float32,[None, n_inputs])
x1 = tf.placeholder(tf.float32,[None, n_inputs])

wx = tf.Variable(tf.random_normal(shape=[n_inputs, n_neurons], dtype=tf.float32))
wy = tf.Variable(tf.random_normal(shape=[n_neurons, n_neurons], dtype=tf.float32))
b = tf.Variable(tf.zeros([1, n_neurons]), dtype=tf.float32)

y0 = tf.tanh(tf.matmul(x0,wx) + b)
y1 = tf.tanh(tf.matmul(y0,wy) + tf.matmul(x1,wx) + b)

init = tf.global_variables_initializer()

import numpy as np
x0_batch = np.array([[0,1,2],[3,4,5],[6,7,8],[9,0,1]])
x1_batch = np.array([[9,8,7],[0,0,0],[6,5,4],[3,2,1]])

with tf.Session() as sess:
    init.run()
    y0_val, y1_val = sess.run([y0,y1],feed_dict={x0: x0_batch, x1: x1_batch})
print(y1_val)
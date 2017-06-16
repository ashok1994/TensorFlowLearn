#A Machine Learning Program to Make the neuron learn to add

# Suppress OS related warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys

# Import tensorflow library
import tensorflow as tf
import numpy as np

sess = tf.Session()


# Inputs - Two Number
n1 = tf.placeholder(tf.float32, name = "num1")
n2 = tf.placeholder(tf.float32, name = "num2")
actual_sum = tf.placeholder(tf.float32, name = "actual_sum")

# Weights for addition
w1 = tf.Variable(0.8, name = "weight1")
w2 = tf.Variable(0.8, name = "weight2")

# The neuron
with tf.name_scope("operation"):
	inp1 = tf.multiply(w1, n1, name = "input1")
	inp2 = tf.multiply(w2, n2, name = "input2")
	y = inp1 + inp2



with tf.name_scope("loss"):
	loss = tf.pow(y - actual_sum , 2, name="loss")

# Training Step : Algorithm -> GradientDescentOptimizer
with tf.name_scope("training"):
    train_step = tf.train.GradientDescentOptimizer(0.00001).minimize(loss)

for value in [n1, n2, actual_sum, loss]:
	tf.summary.scalar(value.op.name, value)


# Merging all summaries : Tensorboard
summaries = tf.summary.merge_all()

# Printing the graph : Tensorboard
summary_writer = tf.summary.FileWriter('log_simple_stats', sess.graph)

# Initialize all variables
sess.run(tf.global_variables_initializer())

for i in range(300):
	sample1 = np.random.uniform(low = 0.0, high = 300.0)
	sample2 = np.random.uniform(low = 0.0, high = 300.0)
	output = np.add(sample1, sample2)

	_, merged = sess.run([train_step, summaries], feed_dict={n1 : sample1, n2 : sample2 , actual_sum : output})
	summary_writer.add_summary(merged, i)
# Output
print(sess.run([w1, w2]))
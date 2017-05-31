# Suppress OS related warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys

# Import tensorflow library
import tensorflow as tf
import numpy as np

sess = tf.Session()


# Input Data X : of placeholder Value 1.0 tf.float32
x = tf.placeholder(tf.float32, name="input")

# Variable Weight : Arbitary Value
w = tf.Variable(0.8, name='weight')

# Neuron : y = w * x
with tf.name_scope('operation'):
    y = tf.multiply(w, x, name='output')

# Actual Output
actual_output = tf.constant(0.0, name="actual_output")

# Loss function , delta square
with tf.name_scope("loss"):
    loss = tf.pow(y - actual_output, 2, name='loss')

# Training Step : Algorithm -> GradientDescentOptimizer
with tf.name_scope("training"):
    train_step = tf.train.GradientDescentOptimizer(0.00001).minimize(loss)

# # Ploting graph : Tensorboard
for value in [x, w, y, actual_output, loss]:
    tf.summary.scalar(value.op.name, value)

# Merging all summaries : Tensorboard
summaries = tf.summary.merge_all()

# Printing the graph : Tensorboard
summary_writer = tf.summary.FileWriter('log_simple_stats', sess.graph)

# Initialize all variables
sess.run(tf.global_variables_initializer())

for i in range(300):
    sample = np.random.uniform(low=0.0, high=300.0)
    _, merged = sess.run([train_step, summaries], feed_dict={x: sample})
    summary_writer.add_summary(merged, i)

# Output
print(sess.run([w]))

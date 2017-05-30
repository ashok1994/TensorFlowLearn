# Suppress OS related warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Import tensorflow library
import tensorflow as tf
sess = tf.Session()

# Input Data X : of Constant Value 1.0 tf.float32
x = tf.constant(1.0, name='input')

# Variable Weight : Arbitary Value
w = tf.Variable(0.8, name='weight')

# Neuron : y = w * x
with tf.name_scope("operation"):
    y = tf.multiply(w, x, name='output')

# Actual Output
y_ = tf.constant(0.0, name='correct_value')

# Loss function , delta square
with tf.name_scope("loss"):
    loss = tf.pow(y - y_, 2, name='loss')

# Training Step : Algorithm -> GradientDescentOptimizer
with tf.name_scope("training"):
    train_step = tf.train.GradientDescentOptimizer(0.025).minimize(loss)

# Ploting graph : Tensorboard
for value in [x, w, y, y_, loss]:
    tf.summary.scalar(value.op.name, value)

# Merging all summaries : Tensorboard
summaries = tf.summary.merge_all()

# Printing the graph : Tensorboard
summary_writer = tf.summary.FileWriter('log_simple_stats', sess.graph)

# Initialize all variables
sess.run(tf.global_variables_initializer())

# Actual Training Loops
for i in range(100):
    summary_writer.add_summary(sess.run(summaries), i)
    sess.run(train_step)

# Output
print(sess.run([w]))

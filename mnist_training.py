import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

save_file = './model.ckpt'

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

x_image = tf.reshape(x, [-1,28,28,1])

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='VALID')


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

kernel_size = 3
databit = 255
precision = 5

conv_filter1 = 8
conv_filter2 = 16

unit1 = 120
unit2 = 84

W_conv1 = weight_variable([kernel_size, kernel_size, 1, conv_filter1])
b_conv1 = bias_variable([8])

#h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_conv1 = conv2d(x_image, W_conv1) + b_conv1
h_conv1 = tf.nn.relu(h_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([kernel_size, kernel_size, conv_filter1, conv_filter2])
b_conv2 = bias_variable([16])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

h_pool2_flat = tf.reshape(h_pool2, [-1, 5*5*conv_filter2])
W_fc1 = weight_variable([5 * 5 * conv_filter2, unit1])
b_fc1 = bias_variable([unit1])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([unit1, unit2])
b_fc2 = bias_variable([unit2])
h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

W_fc3 = weight_variable([unit2, 10])
b_fc3 = bias_variable([10])
y_conv=tf.nn.softmax(tf.matmul(h_fc2, W_fc3) + b_fc3)


cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))

train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()

for i in range(0, 12000):
  batch = mnist.train.next_batch(100)
  if i%100 == 0:

    #train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})

    train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1]})

    #if i % display_step == 0 or i == 1:
    #    saver.save(sess, './save/mdc_session', global_step=i)

    print("step %d, training accuracy %g"%(i, train_accuracy))

  if i%2000 == 0:
      W_conv1_re = sess.run(W_conv1).reshape(9, 8)
      np.savetxt('./mnist_training_weight/W_conv1_int_%d.txt'%(i), W_conv1_re)
      W_conv2_re = sess.run(W_conv2).reshape(9, 128)
      np.savetxt('./mnist_training_weight/W_conv2_int_%d.txt'%(i), W_conv2_re)
      np.savetxt('./mnist_training_weight/W_fc1_int_%d.txt'%(i), sess.run(W_fc1))
      np.savetxt('./mnist_training_weight/W_fc2_int_%d.txt'%(i), sess.run(W_fc2))
      np.savetxt('./mnist_training_weight/W_fc3_int_%d.txt'%(i), sess.run(W_fc3))

  #train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})
saver.save(sess, save_file)
#print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

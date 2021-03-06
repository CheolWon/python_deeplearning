import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
np.random.seed(20160703)
tf.set_random_seed(20160703)

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

num_filters1 = 32

x = tf.placeholder(tf.float32, [None, 784])
x_image = tf.reshape(x, [-1, 28, 28, 1])

W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, num_filters1], stddev=0.1))
b_conv1 = tf.Variable(tf.constant(0.1, shape=[num_filters1]))

h_conv1 = tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME')
h_conv1_cutoff = tf.nn.relu(h_conv1 + b_conv1)

h_pool = tf.nn.max_pool(h_conv1_cutoff, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding='SAME')

h_pool_flat = tf.reshape(h_pool, [-1, 14* 14* num_filters1])

num_units1 = 14 * 14 * num_filters1
num_units2 = 1024

w2 = tf.Variable(tf.truncated_normal([num_units1, num_units2]))
b2 = tf.Variable(tf.zeros([num_units2]))
hidden2 = tf.nn.relu(tf.matmul(h_pool_flat, w2) + b2)

w0 = tf.Variable(tf.zeros([num_units2, 10]))
b0 = tf.Variable(tf.zeros([10]))
p = tf.nn.softmax(tf.matmul(hidden2, w0) + b0)

t = tf.placeholder(tf.float32, [None, 10])
loss = -tf.reduce_sum(t * tf.log(p))
train_step = tf.train.AdamOptimizer(0.0005).minimize(loss)
correct_prediction = tf.equal(tf.argmax(p, 1), tf.argmax(t, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

i = 0
for _ in range(4000):
    i += 1
    batch_xs, batch_ts = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, t: batch_ts})
    if i%100 == 0:
        loss_val, acc_val = sess.run([loss, accuracy], feed_dict={x:mnist.test.images, t:mnist.test.labels})
        print('Step: %d, Loss: %f, Accuracy: %f' %(i, loss_val, acc_val))

print(sess.run(w0))
w0_np = sess.run(w0)
np.savetxt('w0.txt', w0_np)

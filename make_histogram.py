import tensorflow as tf
import numpy as np
import operator
import math
import matplotlib.pyplot as plt
import seaborn as sns
'''
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

sess = tf.InteractiveSession()

A = tf.Variable(tf.truncated_normal((1, 1000), stddev=0.03))
B = tf.Variable(tf.truncated_normal((1000, 10), stddev=0.03))
C = tf.matmul(tf.floor(A*pow(2, 8)), tf.floor(B*pow(2, 7)))
sess.run(tf.global_variables_initializer())

print('A : ', sess.run(A))
print('B : ', sess.run(B))
print('C : ', sess.run(C))

print('matmul C : ', sess.run(C))

C_ = tf.floor(C/pow(2, 7))
print('floor : ', sess.run(C_))

c_r = tf.round(C/pow(2, 7))
print('round : ', sess.run(c_r))


C_softmax = tf.nn.softmax(C_)

print('softmax_C : ', sess.run(C_softmax))
'''

a = 0.25
a_ = a * (pow(2, 5)-1)
print('a_ : ', a_)
a_floor = np.floor(a_)
print('a_floor : ', a_floor)


W_conv1 = np.genfromtxt('./mnist_weights/W_conv1_int.txt')
W_conv1 = np.asarray(W_conv1).astype('int')
#print(W_conv1)
W_conv1 = W_conv1.reshape(72)

W_conv2 = np.genfromtxt('./mnist_weights/W_conv2_int.txt')
W_conv2 = np.asarray(W_conv2).astype('int')
print(W_conv2.shape)
W_conv2 = W_conv2.reshape(9 * 128)
print(W_conv2)

W_fc1 = np.genfromtxt('./mnist_weights/W_fc1_int.txt')
W_fc1 = np.asarray(W_fc1).astype('int')
print(W_fc1.shape)
W_fc1 = W_fc1.reshape(400 * 120)

W_fc2 = np.genfromtxt('./mnist_weights/W_fc2_int.txt')
W_fc2 = np.asarray(W_fc2).astype('int')
print(W_fc2.shape)
W_fc2 = W_fc2.reshape(120 * 84)

W_fc3 = np.genfromtxt('./mnist_weights/W_fc3_int.txt')
W_fc3 = np.asarray(W_fc3).astype('int')
print(W_fc3.shape)
print(W_fc3)
W_fc3 = W_fc3.reshape(84 * 10)

plt.hist(W_conv1, bins = [-16, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
plt.title('W_conv1')
plt.show()
plt.clf()

plt.hist(W_conv2, bins = [-16, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
plt.title('W_conv2')
plt.show()
plt.clf()

plt.hist(W_fc1, bins = [-16, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
plt.title('W_fc1')
plt.show()
plt.clf()


plt.hist(W_fc2, bins = [-16, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
plt.title('W_fc2')
plt.show()
plt.clf()

plt.hist(W_fc3, bins = [-16, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
plt.title('W_fc3')
plt.show()
plt.clf()

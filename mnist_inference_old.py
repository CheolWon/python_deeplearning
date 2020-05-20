import tensorflow as tf
import numpy as np
import operator

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def input_to_integer(x):
    x_int = np.zeros((1, 28, 28, 1))
    for col in range(0, 28):
        for row in range(0, 28):
            x_int[0][col][row][0] = round(x[col][row] * (pow(2, 8)-1))
    return x_int

def resultCompare(a, b):
    result = sess.run(b)
    #print(x_label.argmax())
    #print(result[0].argmax())
    #print(np.equal(a.argmax(), result[0].argmax()))
    return (np.equal(a.argmax(), result[0].argmax()))

conv_filter1 = 8
conv_filter2 = 16
filter_size = 5

unit1 = 120
unit2 = 84

W_conv1 = weight_variable([filter_size, filter_size, 1, conv_filter1])
b_conv1 = bias_variable([conv_filter1])

W_conv2 = weight_variable([filter_size, filter_size, conv_filter1, conv_filter2])
b_conv2 = bias_variable([conv_filter2])

W_fc1 = weight_variable([7 * 7 * conv_filter2, unit1])
b_fc1 = bias_variable([unit1])

W_fc2 = weight_variable([unit1, unit2])
b_fc2 = bias_variable([unit2])

W_fc3 = weight_variable([unit2, 10])
b_fc3 = bias_variable([10])

sess = tf.InteractiveSession()

#가중치 불러오기
#이 때 모든 가중치가 미리 선언되어 있어야 한다
save_file = './model.ckpt'
saver = tf.train.Saver()

saver.restore(sess, save_file)
print('integer')
precision = 6;
#weight W_conv1

#print(sess.run(W_fc1))
W_conv1_np = sess.run(W_conv1)
#print(W_conv1_np)
#np.savetxt("./save_data/W_conv1_np.txt", W_conv1_np)
#print(W_conv1_np.shape)
#print(W_conv1_np[0][0][0][0])
W_conv1_int = np.zeros((filter_size, filter_size, 1, conv_filter1))
#W_conv1_int = round(W_conv1_np[0][0][0][0] * pow(2,4))
#print(W_conv1_int)
num_Count1 = np.zeros(conv_filter1)

for channel in range(0 ,conv_filter1):
    for batch in range(0, 1):
        for col in range(0, filter_size):
            for row in range(0, filter_size):
                W_conv1_int[col][row][batch][channel] = round(W_conv1_np[col][row][batch][channel] * (pow(2, precision)-1))
                #num_Count1[int(W_conv1_int[col][row][batch][channel])+15]+=1

#np.savetxt('W_conv1.txt', W_conv1_int, fmt = '%2x', delimiter=',')

#print(W_conv1_int)
bias_conv1_np = sess.run(b_conv1)
bias_conv1_int = np.zeros(conv_filter1)
for bias in range(0, conv_filter1):
    bias_conv1_int[bias] = round(bias_conv1_np[bias] * (pow(2, precision)-1))
#print(num_Count1)
#np.savetxt('b_conv1.txt', bias_conv1_int, fmt = '%2x', delimiter=',')

#weight conv2
#print(W_conv2)
W_conv2_np = sess.run(W_conv2)
num_Count2 = np.zeros(31)
W_conv2_int = np.zeros((filter_size, filter_size, 1, conv_filter2))

for channel in range(0 ,conv_filter2):
    for batch in range(0, 1):
        for col in range(0, filter_size):
            for row in range(0, filter_size):
                W_conv2_int[col][row][batch][channel] = round(W_conv2_np[col][row][batch][channel] * (pow(2, precision)-1))
                #num_Count2[int(W_conv2_int[col][row][batch][channel])+15]+=1

#np.savetxt('W_conv2.txt', W_conv2_int, fmt = '%2x', delimiter=',')

bias_conv2_np = sess.run(b_conv2)
bias_conv2_int = np.zeros(conv_filter2)
for bias in range(0, conv_filter2):
    bias_conv2_int[bias] = round(bias_conv2_np[bias] * (pow(2, precision)-1))
#print(num_Count2)

#np.savetxt('b_conv2.txt', bias_conv2_int, fmt = '%2x', delimiter=',')
#weight fc1

#print(W_fc1)
W_fc1_np = sess.run(W_fc1)
num_Count3 = np.zeros(31)
num_Count3_1 = np.zeros(16)
W_fc1_int = np.zeros((7*7*conv_filter2, unit1), dtype=int)

for col in range(0, 7*7*conv_filter2):
    for row in range(0, unit1):
        W_fc1_int[col][row] = round(W_fc1_np[col][row] * (pow(2, precision)-1))
        #num_Count3_1[abs(int(W_fc1_int[col][row]))]+=1

np.savetxt('W_fc1.txt', W_fc1_int, fmt = '%3x', delimiter=',')
#print(num_Count3_1)
#bias_fc1

bias_fc1_np = sess.run(b_fc1)
bias_fc1_int = np.zeros(unit1)
for bias in range(0, unit1):
    bias_fc1_int[bias] = round(bias_fc1_np[bias] * (pow(2, precision)-1))
#print(num_Count3_1)

#weight fc2


W_fc2_np = sess.run(W_fc2)
num_Count4 = np.zeros(31)
num_Count4_1 = np.zeros(16)
W_fc2_int = np.zeros((unit1, unit2))

#print(W_fc2_np)

for col in range(0, unit1):
    for row in range(0, unit2):
        W_fc2_int[col][row] = round(W_fc2_np[col][row] * (pow(2, precision)-1))
        #num_Count4_1[abs(int(W_fc2_int[col][row]))]+=1

#print(b_fc2)
#bias_fc2
bias_fc2_np = sess.run(b_fc2)
bias_fc2_int = np.zeros(unit2)
for bias in range(0, unit2):
    bias_fc2_int[bias] = round(bias_fc2_np[bias] * (pow(2, precision)-1))
#print(num_Count4_1)


#weight fc3


W_fc3_np = sess.run(W_fc3)
num_Count4 = np.zeros(31)
num_Count4_1 = np.zeros(16)
W_fc3_int = np.zeros((unit2, 10))

#print(W_fc2_np)

for col in range(0, unit2):
    for row in range(0, 10):
        W_fc3_int[col][row] = round(W_fc3_np[col][row] * (pow(2, precision)-1))
        #num_Count4_1[abs(int(W_fc2_int[col][row]))]+=1

#print(b_fc2)
#bias_fc2
bias_fc3_np = sess.run(b_fc3)
bias_fc3_int = np.zeros(10)
for bias in range(0, 10):
    bias_fc3_int[bias] = round(bias_fc3_np[bias] * (pow(2, precision)-1))
#print(num_Count4_1)

count=0

print('done')
for i in range(100, 200):

    x_image_int = np.zeros((1, 28, 28, 1))

    x_image = mnist.train.images[i]
    x_label = mnist.train.labels[i]

    x_image = x_image.reshape(28, 28)

    #for col in range(0, 28):
    #    for row in range(0, 28):
    #        x_image_int[0][col][row][0] = round(x_image[col][row] * (pow(2, 7)-1))

    x_image_int[0, :, :, 0] = np.floor(((pow(2, 7)-1) * x_image[:, :]))

    h_conv1 = tf.nn.relu(conv2d(x_image_int, W_conv1_int) + bias_conv1_int)
    h_pool1 = max_pool_2x2(h_conv1)
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2_int) + bias_conv2_int)
    h_pool2 = max_pool_2x2(h_conv2)
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*conv_filter2])
    W_fc1_int = W_fc1_int.astype('float64')
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1_int) + bias_fc1_int)
    W_fc2_int = W_fc2_int.astype('float64')

    h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2_int) + bias_fc2_int)

    W_fc3_int = W_fc3_int.astype('float64')
    y_conv = tf.matmul(h_fc2, W_fc3_int) + bias_fc3_int

    result = sess.run(y_conv)

    if(x_label.argmax() == result[0].argmax()):
        count+=1
    print(i)

print('count : ', count)

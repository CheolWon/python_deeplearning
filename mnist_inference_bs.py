import tensorflow as tf
import numpy as np
import operator
import Conv2d as cv2d

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

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

def input_to_integer(x):
    x_int = np.zeros((1, 28, 28, 1))
    for col in range(0, 28):
        for row in range(0, 28):
            x_int[0][col][row][0] = round(x[col][row] * (pow(2, 7)-1))
    return x_int

def scaling_up(a, b):
    c = tf.round(a * (pow(2, b)-1))
    return c

def scaling_down(a, b):
    return tf.floor((a)/(pow(2, shift_right)-1))

kernel_size = 3

conv_filter1 = 8
conv_filter2 = 16

unit1 = 120
unit2 = 84

scale_conv = 5
scale_fc = 5

shift_right = 6
W_conv1 = weight_variable([kernel_size, kernel_size, 1, conv_filter1])
W_conv1 = scaling_up(W_conv1, scale_conv)
b_conv1 = bias_variable([conv_filter1])
b_conv1 = scaling_up(b_conv1, scale_conv)

W_conv2 = weight_variable([kernel_size, kernel_size, conv_filter1, conv_filter2])
W_conv2 = scaling_up(W_conv2, scale_conv)
b_conv2 = bias_variable([conv_filter2])
b_conv2 = scaling_up(b_conv2, scale_conv)

W_fc1 = weight_variable([5 * 5 * conv_filter2, unit1])
W_fc1 = scaling_up(W_fc1, scale_fc)
b_fc1 = bias_variable([unit1])
b_fc1 = scaling_up(b_fc1, scale_fc)

W_fc2 = weight_variable([unit1, unit2])
W_fc2 = scaling_up(W_fc2, scale_fc)
b_fc2 = bias_variable([unit2])
b_fc2 = scaling_up(b_fc2, scale_fc)

W_fc3 = weight_variable([unit2, 10])
W_fc3 = scaling_up(W_fc3, scale_fc)
b_fc3 = bias_variable([10])
b_fc3 = scaling_up(b_fc3, scale_fc)

sess = tf.InteractiveSession()

#가중치 불러오기
#이 때 모든 가중치가 미리 선언되어 있어야 한다
save_file = './model.ckpt'
saver = tf.train.Saver()

saver.restore(sess, save_file)


np_W_conv1 = sess.run(W_conv1)
np_b_conv1 = sess.run(b_conv1)

np_W_conv2 = sess.run(W_conv2)
np_b_conv2 = sess.run(b_conv2)


count = 0
print('floating')
print_featuremap = 0
save_weight = 0
print_weight_channel = 0
for i in range(0, 1000):
    x_image = mnist.test.images[i]
    x_label = mnist.test.labels[i]
    x_image = x_image.reshape(28, 28)

    x_image = x_image[np.newaxis, :, :, np.newaxis]
    '''
    h_conv1 = scaling_down(conv2d(tf.floor(127*x_image), W_conv1) + b_conv1, shift_right)
    h_conv1 = tf.nn.relu(h_conv1)
    h_pool1 = max_pool_2x2(h_conv1)


    h_conv2 = scaling_down(conv2d(h_pool1, W_conv2) + b_conv2, shift_right)
    h_conv2 = tf.nn.relu(h_conv2)
    '''

    #npInImage = sess.run(x_image)

    h_conv1_np = scaling_down(cv2d.conv2d(np.floor(127*x_image), np_W_conv1, np_b_conv1), shift_right)
    h_conv1 = tf.nn.relu(h_conv1_np)
    h_pool1 = max_pool_2x2(h_conv1)

    np_h_pool1 = sess.run(h_pool1)

    h_conv2 = scaling_down(cv2d.conv2d(np_h_pool1, np_W_conv2, np_b_conv2), shift_right)
    h_conv2 = tf.nn.relu(h_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    h_pool2_flat = tf.reshape(h_pool2, [-1, 5*5*conv_filter2])

    #h_pool2_flat_np = sess.run(h_pool2_flat)

    h_fc1 = scaling_down(tf.matmul(h_pool2_flat, W_fc1) + b_fc1, shift_right)
    h_fc1 = tf.nn.relu(h_fc1)

    h_fc2 = scaling_down(tf.matmul(h_fc1, W_fc2) + b_fc2, shift_right)
    h_fc2 = tf.nn.relu(h_fc2)

    y_conv = tf.nn.softmax(tf.matmul(h_fc2, W_fc3) + b_fc3)

    result = sess.run(y_conv)

    if(x_label.argmax() == result[0].argmax()):
        count+=1
    if(i%100==0):
        print('count : ', count)
    #print(i)


    if(save_weight):
        W_conv1_re = sess.run(W_conv1).reshape(9, 8)
        np.savetxt('./mnist_weights/W_conv1_int.txt', W_conv1_re.astype('int'))
        W_conv2_re = sess.run(W_conv2).reshape(9, 128)
        np.savetxt('./mnist_weights/W_conv2_int.txt', W_conv2_re.astype('int'))
        np.savetxt('./mnist_weights/W_fc1_int.txt', sess.run(W_fc1).astype('int'))
        np.savetxt('./mnist_weights/W_fc2_int.txt', sess.run(W_fc2).astype('int'))
        np.savetxt('./mnist_weights/W_fc3_int.txt', sess.run(W_fc3).astype('int'))

    if(print_featuremap):
        print('conv1_floor : ', sess.run(h_conv1))
        print('conv2 : ', sess.run(h_conv2))
        print('fc1 : ', sess.run(h_fc1))
        print('fc2 : ', sess.run(h_fc2))

    if(print_weight_channel):
        print('x_image : ', x_image.shape)
        print('W_conv1 : ', W_conv1)
        print('h_conv1 : ', h_conv1)
        print('h_pool1 : ', h_pool1)
        print('W_conv2 : ', W_conv2)
        print('h_conv2 : ', h_conv2)
        print('h_pool2 : ', h_pool2)

print('count : ', count)

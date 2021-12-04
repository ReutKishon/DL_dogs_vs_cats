import tensorflow.compat.v1 as tf
import pickle
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
tf.disable_v2_behavior()

X_train = pickle.load(open("X_train.pickle", "rb"))/255.0
Y_train = pickle.load(open("y_train.pickle", "rb"))
X_test = pickle.load(open("X_test.pickle", "rb"))/255.0
Y_test = pickle.load(open("y_test.pickle", "rb"))


# image parameters
IMAGE_SIZE = 64
IMAGE_CHANNELS = 1
NUM_CLASSES = 2

x_tensor = tf.placeholder(
    tf.float32, shape=[None, IMAGE_SIZE*IMAGE_SIZE])
y_tensor = tf.placeholder(tf.float32, shape=[None, NUM_CLASSES])

keep_prob = tf.placeholder(tf.float32)


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def cnn():
    x_image = tf.reshape(x_tensor, [-1, IMAGE_SIZE, IMAGE_SIZE, 1])

    W_conv1 = weight_variable([3, 3, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([3, 3, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    flat = tf.reshape(h_pool2, [-1, 16*16*64])
    W_fc1 = weight_variable([16*16*64, 5460])
    b_fc1 = bias_variable([5460])
    h_fc1 = tf.nn.relu(tf.matmul(flat, W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, rate=keep_prob)

    W_fc2 = weight_variable([5460, 2])
    b_fc2 = bias_variable([2])
    y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2
    return y_conv


def next_batch(num):
    '''
    Return a total of `num` random samples and labels.
    '''
    idx_arr = np.arange(0, len(X_train))
    np.random.shuffle(idx_arr)
    idx_arr = idx_arr[:num]  # take the first num samples
    data_shuffle = [X_train[i] for i in idx_arr]
    labels_shuffle = [Y_train[i] for i in idx_arr]

    data = np.asarray(data_shuffle).reshape([num, IMAGE_SIZE*IMAGE_SIZE])

    labels = np.asarray(labels_shuffle)

    return data, labels


def train_and_test():
    y_conv = cnn()

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_tensor, logits=y_conv))
    tf.print(cross_entropy)
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_tensor, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        for i in range(1500):
            batch = next_batch(30)
            labels_one_hot = tf.one_hot(batch[1], 2).eval(
                session=tf.compat.v1.Session())
            if i % 100 == 0:

                train_accuracy = accuracy.eval(feed_dict={
                    x_tensor: batch[0], y_tensor: labels_one_hot, keep_prob: 1.0})
                print('step %d, training accuracy %g' %
                      (i, train_accuracy))

            train_step.run(
                feed_dict={x_tensor: batch[0], y_tensor: labels_one_hot, keep_prob: 0.2})
        test_data = np.asarray(X_test).reshape([2000, IMAGE_SIZE*IMAGE_SIZE])
        test_labels_as_array = np.asarray(Y_test)
        test_labels = tf.one_hot(test_labels_as_array, 2).eval(
            session=tf.compat.v1.Session())

        test_accuracy = accuracy.eval(feed_dict={
            x_tensor: test_data, y_tensor: test_labels, keep_prob: 1.0})
        print('test accuracy %g' % test_accuracy)
    sess.close()


if __name__ == "__main__":

    train_and_test()

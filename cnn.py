# Import libraries
import pickle
import os
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.keras import optimizers
tf.compat.v1.disable_eager_execution()


X = pickle.load(open("X.pickle", "rb"))

Y = pickle.load(open("y.pickle", "rb"))


X = X/255.0


n_classes = 2
batch_size = 128

x = tf.compat.v1.placeholder('float', [None, 784])
y = tf.compat.v1.placeholder('float', shape=(128,))


keep_rate = 0.8
keep_prob = tf.compat.v1.placeholder(tf.float32)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def maxpool2d(x):
    #                        size of window         movement of window
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def convolutional_neural_network(x):

    weights = {'W_conv1': tf.Variable(tf.random.normal([5, 5, 1, 32])),
               'W_conv2': tf.Variable(tf.random.normal([5, 5, 32, 64])),
               'W_fc': tf.Variable(tf.random.normal([7*7*64, 1024])),
               'out': tf.Variable(tf.random.normal([1024, n_classes]))}

    biases = {'b_conv1': tf.Variable(tf.random.normal([32])),
              'b_conv2': tf.Variable(tf.random.normal([64])),
              'b_fc': tf.Variable(tf.random.normal([1024])),
              'out': tf.Variable(tf.random.normal([n_classes]))}
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    conv1 = tf.nn.relu(conv2d(x, weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool2d(conv1)

    conv2 = tf.nn.relu(
        conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool2d(conv2)

    fc = tf.reshape(conv2, [-1, 7*7*64])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc'])+biases['b_fc'])
    fc = tf.nn.dropout(fc, keep_rate)

    output = tf.matmul(fc, weights['out'])+biases['out']

    return output


def next_batch(num):
    '''
    Return a total of `num` random samples and labels.
    '''
    idx_arr = np.arange(0, len(X))
    np.random.shuffle(idx_arr)
    idx_arr = idx_arr[:num]  # take the first num samples
    data_shuffle = [X[i] for i in idx_arr]
    labels_shuffle = [Y[i] for i in idx_arr]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)


# def cost1():
#     prediction = convolutional_neural_network(x=x)
#     print(prediction)

#     return tf.math.reduce_mean(
#         tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))


def train_neural_network():
    prediction = convolutional_neural_network(x=x)
    print(prediction.eval)

    cost = tf.math.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))

    optimizer = tf.train.AdamOptimizer().minimize(loss=cost)

    n_epochs = 10
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(n_epochs):
            epoch_loss = 0
            for _ in range(int(8000/batch_size)):
                epoch_x, epoch_y = next_batch(batch_size)
                print("epoch_x")

                print(epoch_x)
                print("epoch_y")

                print(epoch_y)
                _, c = sess.run([optimizer, cost], feed_dict={
                    x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of',
                  hm_epochs, 'loss:', epoch_loss)

        # correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        # accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        # print('Accuracy:', accuracy.eval(
        #     {x: mnist.test.images, y: mnist.test.labels}))


# train_neural_network()
prediction = convolutional_neural_network(x=x)
# initialize the variable
init_op = tf.initialize_all_variables()

# run the graph
with tf.Session() as sess:
    sess.run(init_op)  # execute init_op
    # print the random values that we sample
    print(sess.run(prediction))

import tensorflow as tf1
import pickle
import numpy as np
tf1.compat.v1.disable_v2_behavior()

tf = tf1.compat.v1
sess = tf.InteractiveSession()
print('Starting...')
X_train = pickle.load(open("X_train.pickle", "rb"))/255.0
Y_train = pickle.load(open("y_train.pickle", "rb"))
X_test = pickle.load(open("X_test.pickle", "rb"))/255.0
Y_test = pickle.load(open("y_test.pickle", "rb"))
print('Data loaded...')


x_tensor = tf.placeholder(tf.float32, shape=[None, 22500])
y_tensor = tf.placeholder(tf.float32, shape=[None, 2])
keep_prob = tf.placeholder(tf.float32)

W = tf.Variable(tf.zeros([22500, 2]))
b = tf.Variable(tf.zeros([2]))

print('Variable initialized...')

sess.run(tf.global_variables_initializer())

# Regression model
y = tf.matmul(x_tensor,W) + b

# Loss function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_tensor, logits=y))

# Initiate the model training
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

def next_batch(num):
    '''
    Return a total of `num` random samples and labels.
    '''
    idx_arr = np.arange(0, len(X_train))
    np.random.shuffle(idx_arr)
    idx_arr = idx_arr[:num]  # take the first num samples
    data_shuffle = [X_train[i] for i in idx_arr]
    labels_shuffle = [Y_train[i] for i in idx_arr]

    data = np.asarray(data_shuffle)

    data = data.reshape(num, 22500)
    labels = np.asarray(labels_shuffle)

    return data, labels

print('Start iterating...')

# Train the model by iterating the train fnuction
for _ in range(5000):
    batch = next_batch(100)
    labels_one_hot = tf.one_hot(batch[1], 2).eval(session=tf.Session())
    train_step.run(feed_dict={x_tensor: batch[0], y_tensor: labels_one_hot})

# Check our accuracy
correct_prediction = tf.equal(tf.argmax(y,1), tf.arg_max(y_tensor,1))

# Calculate the date of accuracy that we have got
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
test_data = np.asarray(X_test).reshape(2000, 22500)
test_labels_as_array = np.asarray(Y_test)
test_labels = tf.one_hot(test_labels_as_array, 2).eval(
            session=tf.compat.v1.Session())
print(accuracy.eval(feed_dict={x_tensor: test_data, y_tensor: test_labels}))

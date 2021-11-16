import tensorflow as tf


def get_train():

    train_ds = tf.keras.utils.image_dataset_from_directory(
        'dataset/training_set/',
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    return train_ds


def conv2d(x, w):
    weights = tf.Variable(tf.random.normal(w))
    return tf.nn.relu(tf.nn.conv2d(x, weights, [1, 1, 1, 1], padding='SAME'))


def dropout(x, prob):
    return tf.nn.dropout(x, prob)


def max_pool(x, k):
    return tf.nn.max_pool(x, k, [1, 2, 2, 1], padding='SAME')


def fully_connected(x, size):
    weights = tf.Variable(tf.random.normal(
        [int(x.get_shape()[1]), size], mean=0, stddev=1))
    return tf.matmul(x, weights)


def cost_func(x, y):
    cost = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=x)
    cost_1 = tf.reduce_mean(cost)
    return cost_1


if __name__ == "__main__":

    tf.reset_default_graph()
    inp = tf.placeholder(tf.float32, shape=(None, 28, 28, 1), name='input')
    target = tf.placeholder(tf.float32, shape=(None, 10), name='target')
    l1 = conv2d(inp, [5, 5, 1, 32])
    l2 = max_pool(l1, [1, 2, 2, 1])
    l9 = dropout(l2, 0.9)
    l3 = conv2d(l9, [5, 5, 32, 64])
    l4 = max_pool(l3, [1, 2, 2, 1])
    l0 = dropout(l4, 0.9)
    l5 = tf.reshape(l0, [-1, 7*7*64])
    l6 = fully_connected(l5, 10)
    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=target, logits=l6))
    opt = tf.train.AdamOptimizer(0.005).minimize(cost)
    prediction = tf.argmax(l6, 1)
    result = tf.equal(tf.argmax(l6, 1), tf.argmax(target, 1))
    accuracy = tf.reduce_mean(tf.cast(result, tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
    for q in range(5):
        z = train_gen()
        print("EPOCH="+str(q))
        for x, y in z:
            a, b, c = sess.run([cost, opt, accuracy], feed_dict={
                               'input:0': x, 'target:0': y})
        result = sess.run([cost, accuracy], feed_dict={'input:0': np.array(
            xtest).reshape(-1, 28, 28, 1), 'target:0': np.array(ztest).reshape(-1, 10)})
        print("Epoch "+str(q)+" accuracy" +
              str(result[1])+" loss="+str(result[0]))
    f = sess.run([prediction], feed_dict={'input:0': tester})


def train_gen():
    image = []
    label = []
    count = 0
    for x, y in zip(xtrain, ztrain):
        if count < 8:
            count += 1
            image.append(x)
            label.append(y)
        if count == 8:
            yield np.array(image).reshape(-1, 28, 28, 1), np.array(label).reshape(-1, 10)
            count = 0
            image = []
            label = []


def val_gen():
    image = []
    label = []
    count = 0
    for x, y in zip(xtest, ztest):
        if count < 64:
            count += 1
            image.append(x)
            label.append(y)
        if count == 64:
            yield np.array(image).reshape(-1, 28, 28, 1), np.array(label).reshape(-1, 10)
            count = 0
            image = []
            label = []

import numpy as np
import random
import cv2
import os
from math import ceil

ROWS = 64
COLS = 64
CHANNELS = 3

train_dir_cats = "dataset/training_set/Cat/"
test_dir_cats = "dataset/test_set/Cat/"

train_images_cats = [train_dir_cats+i for i in os.listdir(train_dir_cats)]
test_images_cats = [test_dir_cats+i for i in os.listdir(test_dir_cats)]

train_dir_dogs = "dataset/training_set/Dog/"
test_dir_dogs = "dataset/test_set/Dog/"

train_images_dogs = [train_dir_dogs+i for i in os.listdir(train_dir_dogs)]
test_images_dogs = [test_dir_dogs+i for i in os.listdir(test_dir_dogs)]

train_images = train_images_cats + train_images_dogs
test_images = test_images_cats + test_images_dogs

# train_images.remove('dataset/training_set/Dog/_DS_Store')
# train_images.remove('dataset/training_set/Cat/_DS_Store')
# test_images.remove('dataset/test_set/Dog/_DS_Store')
# test_images.remove('dataset/test_set/Cat/_DS_Store')
random.shuffle(train_images)
random.shuffle(test_images)

print(len(train_images))
print(len(test_images))


def read_image(file_path):
    if file_path.endswith('.jpg'):
        img = cv2.imread(file_path, 1)
    return cv2.resize(img, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)


def prepare_data(images):

    m = len(images)
    X = np.zeros((m, ROWS, COLS, CHANNELS), dtype=np.uint8)
    y = np.zeros((1, m))
    for i, image_file in enumerate(images):
        # print(image_file)
        X[i, :] = read_image(image_file)
        if 'dog.' in image_file.lower():
            # print("dog")
            y[0, i] = 1
        elif 'cat.' in image_file.lower():
            y[0, i] = 0
            # print("cat")
    return X, y


train_set_x, train_set_y = prepare_data(train_images)
test_set_x, test_set_y = prepare_data(test_images)

train_set_x_flatten = train_set_x.reshape(
    train_set_x.shape[0], CHANNELS*COLS*ROWS).T
test_set_x_flatten = test_set_x.reshape(test_set_x.shape[0], -1).T
test_set_x = test_set_x_flatten/255
train_set_x = train_set_x_flatten/255


def sigmoid(z):
    s = 1/(1+np.exp(-z))
    return s


def initialize_with_zeros(dim):

    w = np.zeros((dim, 1))
    b = 0

    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))

    return w, b


def propagate(w, b, X, Y):

    m = X.shape[1]

    A = sigmoid(np.dot(w.T, X) + b)

    cost = (-1.0) * np.mean(np.multiply(Y, np.log(A)) + np.multiply(1.0-Y,
                                                                    np.log(1.0 - A)), axis=1)                                 # compute cost

    dw = (1/m)*np.dot(X, np.transpose(A-Y))
    db = np.average((A-Y))

    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())

    grads = {"dw": dw,
             "db": db}

    return grads, cost


w = np.array([[1.], [2.]])
b = 4.
X = np.array([[5., 6., -7.], [8., 9., -10.]])
Y = np.array([[1, 0, 1]])

grads, cost = propagate(w, b, X, Y)
print("dw = " + str(grads["dw"]))
print("db = " + str(grads["db"]))
print("cost = " + str(cost))


def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    costs = []

    for i in range(num_iterations):

        grads, cost = propagate(w, b, X, Y)

        dw = grads["dw"]
        db = grads["db"]

        w = w - learning_rate*dw
        b = b - learning_rate*db

        if i % 100 == 0:
            costs.append(cost)

        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    params = {"w": w,
              "b": b}

    grads = {"dw": dw,
             "db": db}

    return params, grads, costs


params, grads, costs = optimize(
    w, b, X, Y, num_iterations=100, learning_rate=0.002, print_cost=False)

print("w = " + str(params["w"]))
print("b = " + str(params["b"]))
print("dw = " + str(grads["dw"]))
print("db = " + str(grads["db"]))


def predict(w, b, X):
    '''
    '''

    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    A = sigmoid(np.dot(w.T, X)+b)

    for i in range(A.shape[1]):

        if A[0, i] > 0.5:
            Y_prediction[0, i] = 1
        else:
            Y_prediction[0, i] = 0

    assert(Y_prediction.shape == (1, m))

    return Y_prediction


def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.003, print_cost=False):

    w, b = initialize_with_zeros(X_train.shape[0])

    parameters, grads, costs = optimize(
        w, b, X_train, Y_train, num_iterations, learning_rate, print_cost=print_cost)

    w = parameters["w"]
    b = parameters["b"]

    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    print("train accuracy: {} %".format(
        100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(
        100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}

    return d


def find_opt_alpha(train_set_x, train_set_y, test_set_x, test_set_y):
    learning_rate = 0.006
    const_decrease = 0.001
    solutions = []

    for i in range(5):
        learning_rate -= const_decrease
        print("-----------------------------------------")
        print("learning rate: {} %".format(learning_rate))
        solutions.append(model(train_set_x, train_set_y, test_set_x, test_set_y,
                         num_iterations=500, learning_rate=learning_rate, print_cost=True))

    solutions.sort(key=lambda x: x['learning_rate'])
    return solutions[0]['learning_rate']


d = model(train_set_x, train_set_y, test_set_x, test_set_y,
          num_iterations=10000, learning_rate=0.002, print_cost=True)

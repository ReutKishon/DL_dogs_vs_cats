import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
import random


DATADIR_TRAIN = "./dataset/training_set"
DATADIR_TEST = "./dataset/test_set"

CATEGORIES = ["Dog", "Cat"]


training_data = []
testing_data = []

IMG_SIZE = 150


def create_training_data():
    for category in CATEGORIES:  # do dogs and cats

        # create path to dogs and cats
        path = os.path.join(DATADIR_TRAIN, category)
        # get the classification  (0 or a 1). 0=dog 1=cat
        class_num = CATEGORIES.index(category)

        for img in tqdm(os.listdir(path)):  # iterate over each image per dogs and cats
            try:
                img_array = cv2.imread(os.path.join(
                    path, img), cv2.IMREAD_GRAYSCALE)  # convert to array
                # resize to normalize data size
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                # add this to our training_data
                training_data.append([new_array, class_num])
            except Exception as e:  # in the interest in keeping the output clean...
                pass


def create_testing_data():
    for category in CATEGORIES:  # do dogs and cats

        # create path to dogs and cats
        path = os.path.join(DATADIR_TEST, category)
        # get the classification  (0 or a 1). 0=dog 1=cat
        class_num = CATEGORIES.index(category)

        for img in tqdm(os.listdir(path)):  # iterate over each image per dogs and cats
            try:
                img_array = cv2.imread(os.path.join(
                    path, img), cv2.IMREAD_GRAYSCALE)  # convert to array

                # resize to normalize data size
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                # plt.imshow(new_array, cmap='gray')
                # plt.show()
                # add this to our training_data
                testing_data.append([new_array, class_num])
            except Exception as e:  # in the interest in keeping the output clean...
                pass


create_training_data()
create_testing_data()
random.shuffle(training_data)
random.shuffle(testing_data)
X_train = []
y_train = []
X_test = []
y_test = []

for features, label in training_data:
    X_train.append(features)
    y_train.append(label)

for features, label in testing_data:
    
    X_test.append(features)
    y_test.append(label)

X_train = np.array(X_train).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
X_test = np.array(X_test).reshape(-1, IMG_SIZE, IMG_SIZE, 1)


pickle_out = open("X_train.pickle", "wb")
pickle.dump(X_train, pickle_out)
pickle_out.close()

pickle_out = open("y_train.pickle", "wb")
pickle.dump(y_train, pickle_out)
pickle_out.close()


pickle_out = open("X_test.pickle", "wb")
pickle.dump(X_test, pickle_out)
pickle_out.close()

pickle_out = open("y_test.pickle", "wb")
pickle.dump(y_test, pickle_out)
pickle_out.close()

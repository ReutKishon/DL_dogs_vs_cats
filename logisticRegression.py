import os
import shutil
import cv2 as cv2
from random import shuffle
# import matplotlib.pyplot as plt
import numpy as np
import tqdm as tqdm
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()

TRAIN_DIR= './input_from_kaggle/train'
TEST_DIR = './test1'
IMG_SIZE = 50

def label_img(img): 
    word_label = img.split('.')[-3]
    # [cat, not dog]
    if word_label == 'cat': return 1
    # [not cat, dog]
    elif word_label == 'dog': return 0

def create_train_data(): 
    data_x = []
    data_y = []
    for img in os.listdir(TRAIN_DIR): 
        y = label_img(img)
        img_path = os.path.join(TRAIN_DIR, img)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        data_x.append(np.array(img))
        data_y.append(y)
    return (data_x, data_y)

def image_to_vector(img_path):
    data_x = []
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
    data_x.append(np.array(img))
    return data_x


# def process_test_data(): 
#     testing_data = [] 
#     for img in tqdm(os.listdir(TEST_DIR)): 
#         path = os.path.join(TEST_DIR,img)
#         img_num = img.split('.')[0]
#         img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
#         img = cv2.resize(img, (IMG_SIZE,IMG_SIZE)) 
#         testing_data.append([np.array(img), img_num])
#     shuffle(testing_data)
#     np.save('test_data.npy', testing_data)
#     return testing_data

# Here i need to extract the data as a features


#This is the logistic function:
def h(x,w,b):
    return 1/(1+np.exp(-(np.dot(x,w) + b)))

data_x_y = create_train_data()
data_x_flatten = np.array([i.flatten() for i in data_x_y[0]])

w_1 = [0.]
w_2 = [0]*2499
w = np.append(w_1, w_2)
b = 0
alpha = 0.001
# [cat, dog]
for iteration in range(100):
    gradient_b = np.mean(1*((h(data_x_flatten,w,b))-data_x_y[1]))
    gradient_w = np.dot((h(data_x_flatten,w,b)-data_x_y[1]), data_x_flatten)*1/len(data_x_y[1])
    b -= alpha*gradient_b
    w -= alpha*gradient_w

print(w,b)

iter=0
imgs = []
for img in os.listdir(TRAIN_DIR):
    imgs.append(img)
    y = label_img(img)
    img_path = os.path.join(TRAIN_DIR, img)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
    iter+=1
    if(iter==3):
        break

data_x_1 = image_to_vector(os.path.join(TRAIN_DIR, imgs[0]))
data_x_2 = image_to_vector(os.path.join(TRAIN_DIR, imgs[1]))
data_x_3 = image_to_vector(os.path.join(TRAIN_DIR, imgs[2]))
print("Test image 1 is should be: ", h(np.array(data_x_1).flatten(),w,b))
print("Test image 2 is should be: ", h(np.array(data_x_2).flatten(),w,b))
print("Test image 3 is should be: ", h(np.array(data_x_3).flatten(),w,b))
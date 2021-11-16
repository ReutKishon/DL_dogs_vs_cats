# Import libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

if __name__ == "__main__":
    batch_size = 32
    img_height = 28
    img_width = 28

    train_ds = tf.keras.utils.image_dataset_from_directory(
        'dataset/training_set/',
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    val_ds = tf.keras.utils.image_dataset_from_directory(
        'dataset/training_set/',
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    class_names = train_ds.class_names

    print(class_names[0])
    images = train_ds.take(1)
    y = [y for x, y in train_ds]
    print(y)


# print(class_names_val)

# plt.figure(figsize=(10, 10))
# for images, labels in train_ds.take(1):
#     for i in range(9):
#         ax = plt.subplot(3, 3, i + 1)
#         plt.imshow(images[i].numpy().astype("uint8"))
#         plt.title(class_names[labels[i]])
#         plt.axis("off")
#         plt.show()

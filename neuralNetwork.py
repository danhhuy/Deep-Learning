import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.compat.v2.config.experimental.set_memory_growth(gpu, True)
    
#tải folder lên để train
batch_size = 32
train_data = tf.keras.utils.image_dataset_from_directory(
    'D:/ImageProcessing/deepLearning/MNIST_data/folder1/',
    image_size=(64,64),
    validation_split=0.2,
    labels='inferred', label_mode='int',
    subset="training",
    seed=123,
    batch_size=batch_size)

test_data = tf.keras.utils.image_dataset_from_directory(
    'D:/ImageProcessing/deepLearning/MNIST_data/folder1/',
    image_size=(64,64),
    validation_split=0.2,
    labels='inferred', label_mode='int',
    subset="validation",
    seed=123,
    batch_size=batch_size)

train_ds = train_data.prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = test_data.prefetch(buffer_size=tf.data.AUTOTUNE)
#chuan hoa du lieu tu [0,255] ve [0,1]

#AUTOTUNE = tf.data.AUTOTUNE
#train_data = train_data.cache().prefetch(buffer_size=AUTOTUNE)
#test_data = test_data.cache().prefetch(buffer_size=AUTOTUNE)
#build model
model = tf.keras.models.Sequential(
    [tf.keras.layers.Conv2D(32, [3, 3], activation='relu', input_shape=(64,64,3)),
    tf.keras.layers.Conv2D(64, [3, 3], activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10),
    ])
#bien dich model
model.compile(loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),optimizer="adam", metrics=["accuracy"]) #tinh do chinh xac giua predict va thuc te

#train model
H = model.fit(train_data , validation_data=test_data ,epochs=1)

prediction = model.predict(test_data[0])
print(prediction)
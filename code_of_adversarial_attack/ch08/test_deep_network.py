import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets, metrics


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16,3, (1,1), activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.Conv2D(16,3, (1,1), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2),2),
    tf.keras.layers.Conv2D(32,3, (1,1), activation='relu'),
    tf.keras.layers.Conv2D(32,3, (1,1), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2),2),
    tf.keras.layers.Conv2D(64,3, (1,1), activation='relu'),
    tf.keras.layers.Conv2D(64,3, (1,1), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2),2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.summary()
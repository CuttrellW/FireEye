import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib

DATASET_PATH = './datasets'


class ImageClassifier:

    def __init__(self):
        self.data_dir = pathlib.Path(DATASET_PATH)
        self.fire = list(self.data_dir.glob('fire-images/*'))
        self.no_fire = list(self.data_dir.glob('non-fire-images/*'))
        self.image_width = -1
        self.image_height = -1
        self.epochs = 10
        self.train_ds = tf.keras.Dataset()
        self.validate_ds = tf.keras.Dataset()
        self.model = tf.keras.Sequential()

    def set_image_width(self, width):
        self.image_width = width

    def set_image_height(self, height):
        self.image_height = height

    def load_dataset(self, subset, split=0.2, batch_size=32):
        ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.data_dir,
            subset=subset,
            validation_split=split,
            image_size=(self.image_height, self.image_width),
            batch_size=batch_size
        )

        return ds

    def optimize_dataset_performance(self, dataset, prefetch=True):
        num_items = dataset.cardinality().numpy()
        for ds in [self.train_ds, self.validate_ds]:
            ds = ds.cache().shuffle(num_items/2)
            if prefetch:
                ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    def create_model(self, augmentation=False, dropout=-1):
        sequential_params = []

        if augmentation:
            data_augmentation = keras.Sequential(
                [
                    layers.RandomFlip("horizontal", input_shape=(self.image_height, self.image_width, 3)),
                    layers.RandomRotation(0.1),
                    layers.RandomZoom(0.1),
                ]
            )
            sequential_params.extend([data_augmentation, layers.Rescaling(1. / 255)])
        else:
            sequential_params.append(layers.Rescaling(1. / 255, input_shape=(self.image_height, self.image_width, 3)))

        if dropout != -1:
            sequential_params.append(layers.Dropout(dropout))

        sequential_params.extend([
            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(len(self.train_ds.class_names))
        ])

        model = Sequential(sequential_params)

        return model

    def compile_model(self, model, metrics, summarize=True):
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=metrics)
        if summarize:
            model.summarize()

    def train_model(self):

        history = self.model.fit(
          self.train_ds,
          validation_data=self.validate_ds,
          epochs=self.epochs
        )

        return history


if __name__ == '__main__':
    ic = ImageClassifier()

    history = ic.train_model()

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(ic.epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()
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

    def __init__(self, config):
        self.parameters = config
        self.data_dir = pathlib.Path(DATASET_PATH)
        self.dataset_config = self.parameters['dataset']
        self.model_config = self.parameters['model']
        self.image_width = self.parameters['image']['width']
        self.image_height = self.parameters['image']['height']
        self.epochs = self.parameters['epochs']

    def load_dataset(self, subset):
        ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.data_dir,
            subset=subset.lower(),
            seed=self.dataset_config['seed'],
            validation_split=self.dataset_config['split'],
            image_size=(self.image_height, self.image_width),
            batch_size=self.dataset_config['batchSize']
        )

        shuffle = self.dataset_config[subset.lower()]['shuffle']
        prefetch = self.dataset_config[subset.lower()]['shuffle']

        ds = ds.cache()
        if shuffle != -1:
            ds.shuffle(shuffle)
        if prefetch:
            ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)

        return ds

    def create_model(self):
        sequential_params = []
        rescaling = self.model_config['rescaling']
        dropout = self.model_config['dropout']

        if self.model_config['useAugmentation']:
            augmentation = self.model_config['augmentation']
            data_augmentation = keras.Sequential(
                [
                    layers.RandomFlip(augmentation['randomFlip'], input_shape=(self.image_height, self.image_width, 3)),
                    layers.RandomRotation(augmentation['randomRotation']),
                    layers.RandomZoom(augmentation['randomZoom']),
                ]
            )
            sequential_params.extend([data_augmentation, layers.Rescaling(rescaling)])
        else:
            sequential_params.append(layers.Rescaling(rescaling, input_shape=(self.image_height, self.image_width, 3)))

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
            layers.Dense(self.model_config['layerUnits'], activation='relu'),
            layers.Dense(2)
        ])

        model = Sequential(sequential_params)

        compile_config = self.model_config['compile']

        model.compile(optimizer=compile_config['optimizer'],
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=compile_config['metrics'])

        return model

    def train_model(self, model, train_dataset, validate_dataset):

        history = model.fit(
            train_dataset,
            validation_data=validate_dataset,
            epochs=self.epochs
        )

        return history

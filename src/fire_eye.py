import time

import matplotlib.pyplot as plt
import matplotlib
import yaml
from image_classifier import ImageClassifier
from detection import *

CONFIG_PATH = 'config/default.yaml'


def load_config_file(filename):
    with open(filename, 'r') as stream:
        try:
            parsed_yaml = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    return parsed_yaml


def plot_history(history, epochs_range):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

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


if __name__ == '__main__':
    config = load_config_file(CONFIG_PATH)
    parameters = config['parameters']
    ic = ImageClassifier(parameters)

    # Load Training and Validation Datasets
    training_dataset = ic.load_dataset("training")
    validation_dataset = ic.load_dataset("validation")

    # Create and Compile Model
    model = ic.create_model()
    model.summary()

    # Train Model and Capture Result Metrics
    data = ic.train_model(model, training_dataset, validation_dataset)

    selection = 0
    while selection != '3':
        print("\nPlease choose an option:\n1) Begin Watch\n2) Plot training data\n3) Exit program\n")
        selection = input("Type option # and press enter:\n")
        if selection == '1':
            # Begin watch for fire
            alert = begin_watch(model, parameters['watch'])
            # Alert if fire detected
            if alert:
                alert(config['alert'])
        elif selection == '2':
            # Plot data
            plot_history(data, range(ic.epochs))
        elif selection == '3':
            pass
        else:
            print("Selection not recognized. Please try again\n")
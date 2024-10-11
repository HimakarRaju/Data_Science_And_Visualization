import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def load_cifar10_data(data_dir='cifar-10-batches-py'):
    print("Loading CIFAR-10 dataset...")

    x_train = []
    y_train = []

    # Load training batches
    for i in range(1, 6):
        with open(os.path.join(data_dir, f'data_batch_{i}'), 'rb') as f:
            print(f"Processing data_batch_{i}...")
            batch = pickle.load(f, encoding='bytes')
            x_train.append(batch[b'data'])  # Use b'data'
            y_train.append(batch[b'labels'])

    x_train = np.concatenate(x_train)
    y_train = np.concatenate(y_train)

    # Load test batch
    with open(os.path.join(data_dir, 'test_batch'), 'rb') as f:
        print("Processing test_batch...")
        test_batch = pickle.load(f, encoding='bytes')
        x_test = test_batch[b'data']  # Use b'data'
        y_test = test_batch[b'labels']

    # Reshape and normalize the data
    x_train = x_train.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1) / 255.0
    x_test = x_test.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1) / 255.0

    print("Dataset loaded successfully.")
    return (x_train, y_train), (x_test, y_test)


def build_model(input_shape):
    print("Building the model...")
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    print("Model built successfully.")
    return model


def display_sample_images(x, y):
    print("Displaying sample images...")
    plt.figure(figsize=(10, 10))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(x[i])
        plt.title(f'Label: {y[i]}')
        plt.axis('off')
    plt.show()


def main():
    # Load the CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = load_cifar10_data()

    # Display sample images from the training set
    display_sample_images(x_train, y_train)

    # Build the model
    model = build_model(input_shape=(32, 32, 3))

    # Train the model
    print("Training the model...")
    model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

    # Evaluate the model
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f'\nTest accuracy: {test_acc}')


if __name__ == "__main__":
    main()



import pickle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from .utils import get_shape



def load_mnist_dataset(num_samples_per_class=None):
    # Load the MNIST dataset
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

    # Normalize the images to the range [0, 1]
    train_images = train_images.astype('float32') / 255.0
    test_images = test_images.astype('float32') / 255.0

    # Convert images to a 3D list
    train_images_3d = [image.tolist() for image in train_images]
    test_images_3d = [image.tolist() for image in test_images]

    # Convert labels to one-hot encoded 1D list
    train_labels_one_hot = tf.keras.utils.to_categorical(train_labels, num_classes=10)
    test_labels_one_hot = tf.keras.utils.to_categorical(test_labels, num_classes=10)

    # Convert one-hot encoded labels to lists of lists
    train_labels_1d = [label.tolist() for label in train_labels_one_hot]
    test_labels_1d = [label.tolist() for label in test_labels_one_hot]

    val_images_3d = []
    val_labels_1d = []

    # Optionally, select a balanced subset of training data
    if num_samples_per_class is not None:
        indices = []
        test_indices = []
        val_indices = []
        for i in range(10):
            indices += np.where(train_labels == i)[0][:num_samples_per_class].tolist()
            val_indices += np.where(train_labels == i)[0][num_samples_per_class:int(num_samples_per_class*1.1)].tolist()
            test_indices += np.where(test_labels == i)[0][:int(num_samples_per_class*0.2)].tolist()

        val_images_3d = [train_images_3d[i] for i in val_indices]
        val_labels_1d = [train_labels_1d[i] for i in val_indices]
        train_images_3d = [train_images_3d[i] for i in indices]
        train_labels_1d = [train_labels_1d[i] for i in indices]
        test_images_3d = [test_images_3d[i] for i in test_indices]
        test_labels_1d = [test_labels_1d[i] for i in test_indices]

    # Reshape the images to (batch_size, 1, 28, 28)
    train_images_reshaped = np.reshape(train_images_3d, (len(train_images_3d), 1, 28, 28)).tolist()
    val_images_reshaped = np.reshape(val_images_3d, (len(val_images_3d), 1, 28, 28)).tolist()
    test_images_reshaped = np.reshape(test_images_3d, (len(test_images_3d), 1, 28, 28)).tolist()
    print(f'Loaded {len(train_images_reshaped)} training images, {len(test_images_reshaped)} test images, and {len(val_images_reshaped)} validation images')
    return train_images_reshaped, train_labels_1d, test_images_reshaped, test_labels_1d, val_images_reshaped, val_labels_1d

def load_catdog_dataset(num_samples_per_class=None):
    train_path = "training_set/training_set"
    test_path = 'test_set/test_set'

    train_batches = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input) \
        .flow_from_directory(directory=train_path, target_size=(50, 50), classes=['cats', 'dogs'], batch_size=10)
    test_batches = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input) \
        .flow_from_directory(directory=test_path, target_size=(50, 50), classes=['cats', 'dogs'], batch_size=10, shuffle=False)

    train_images = []
    train_labels = []
    for i in range(len(train_batches)):
        X, y = next(train_batches)
        train_images.append(X)
        train_labels.append(y)

    test_images = []
    test_labels = []
    for i in range(len(test_batches)):
        X, y = next(test_batches)
        test_images.append(X)
        test_labels.append(y)

    train_images = np.vstack(train_images)
    train_labels = np.vstack(train_labels)
    test_images = np.vstack(test_images)
    test_labels = np.vstack(test_labels)

    train_images_reshaped = train_images.transpose((0, 3, 1, 2)).tolist()
    test_images_reshaped = test_images.transpose((0, 3, 1, 2)).tolist()

    train_labels_1d = [[int(np.argmax(label))] for label in train_labels]
    test_labels_1d = [[int(np.argmax(label))] for label in test_labels]

    val_images_reshaped = []
    val_labels_1d = []

    if num_samples_per_class is not None:
        indices = []
        test_indices = []
        val_indices = []
        for i in range(2):
            class_indices = np.where(np.argmax(train_labels, axis=1) == i)[0]
            test_i = np.where(np.argmax(test_labels, axis=1) == i)[0]
            indices += class_indices[:num_samples_per_class].tolist()
            val_indices += class_indices[num_samples_per_class:int(num_samples_per_class * 1.1)].tolist()
            test_indices += test_i[:int(num_samples_per_class*0.2)].tolist()

        val_images_reshaped = [train_images_reshaped[i] for i in val_indices]
        val_labels_1d = [train_labels_1d[i] for i in val_indices]
        train_images_reshaped = [train_images_reshaped[i] for i in indices]
        train_labels_1d = [train_labels_1d[i] for i in indices]
        test_images_reshaped = [test_images_reshaped[i] for i in test_indices]
        test_labels_1d = [test_labels_1d[i] for i in test_indices]

    print(f'Loaded {len(train_images_reshaped)} training images, {len(test_images_reshaped)} test images, and {len(val_images_reshaped)} validation images')
    return train_images_reshaped, train_labels_1d, test_images_reshaped, test_labels_1d, val_images_reshaped, val_labels_1d

def display_images(images, labels, num_images=5):
    fig, axes = plt.subplots(1, num_images, figsize=(10, 3))
    for i in range(num_images):
        if len(images[i]) == 1:

            image = images[i][0]
            cmap = 'gray'
        elif len(images[i]) == 3:
            image = np.transpose(images[i], (1, 2, 0))
            cmap = None
        else:
            raise ValueError("Unsupported image shape: {}".format(get_shape(images[i])))
        
        axes[i].imshow(image, cmap=cmap)
        # Convert one-hot encoded label back to integer label
        label = np.argmax(labels[i]) if isinstance(labels[i], (list)) and len(labels[i]) > 1 else labels[i]
        axes[i].set_title(f'Label: {label}')
        axes[i].axis('off')
    plt.show()

def load_fast(file_path):
    with open(file_path, 'rb') as f:
        loaded_data = pickle.load(f)
        train_images = loaded_data['train_images']
        train_labels = loaded_data['train_labels']
        test_images = loaded_data['test_images']
        test_labels = loaded_data['test_labels']
        val_images = loaded_data['val_images']
        val_labels = loaded_data['val_labels']
    return train_images, train_labels, test_images, test_labels , val_images, val_labels
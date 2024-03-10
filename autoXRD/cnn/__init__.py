import numpy as np
import tensorflow as tf
from random import shuffle
from tensorflow.keras import regularizers
import sys

# Used to apply dropout during training *and* inference
class CustomDropout(tf.keras.layers.Layer):

    def __init__(self, rate, **kwargs):
        super(CustomDropout, self).__init__(**kwargs)
        self.rate = rate

    def get_config(self):
        config = super().get_config()
        config.update({
            "rate": self.rate
        })
        return config

    # Always apply dropout
    def call(self, inputs, training=None):
        return tf.nn.dropout(inputs, rate=self.rate)


class DataSetUp(object):
    """
    Class used to train a convolutional neural network on a given
    set of X-ray diffraction spectra to perform phase identification.
    """

    def __init__(self, xrd, testing_fraction=0):
        """
        Args:
            xrd: a numpy array containing xrd spectra categorized by
                their associated reference phase.
                The shape of the array should be NxMx4501x1 where:
                N = the number of reference phases,
                M = the number of augmented spectra per reference phase,
                4501 = intensities as a function of 2-theta
                (spanning from 10 to 80 degrees by default)
            testing_fraction: fraction of data (xrd patterns) to reserve for testing.
                By default, all spectra will be used for training.
        """
        self.xrd = xrd
        self.testing_fraction = testing_fraction
        self.num_phases = len(xrd)

    @property
    def phase_indices(self):
        """
        List of indices to keep track of xrd spectra such that
            each index is associated with a reference phase.
        """
        xrd = self.xrd
        num_phases = self.num_phases
        return [v for v in range(num_phases)]

    @property
    def x(self):
        """
        Feature matrix (array of intensities used for training)
        """
        intensities = []
        xrd = self.xrd
        phase_indices = self.phase_indices
        for (augmented_spectra, index) in zip(xrd, phase_indices):
            for pattern in augmented_spectra:
                intensities.append(pattern)
        return np.array(intensities)

    @property
    def y(self):
        """
        Target property to predict (one-hot encoded vectors associated
        with the reference phases)
        """
        xrd = self.xrd
        phase_indices = self.phase_indices
        one_hot_vectors = []
        for (augmented_spectra, index) in zip(xrd, phase_indices):
            for pattern in augmented_spectra:
                assigned_vec = [0]*len(xrd)
                assigned_vec[index] = 1.0
                one_hot_vectors.append(assigned_vec)
        return np.array(one_hot_vectors)

    def split_training_testing(self):
        """
        Training and testing data will be split according
        to self.testing_fraction

        Returns:
            x_train, x_test: features matrices (xrd spectra) to be
                used for training and testing
            y_train, t_test: target properties (one-hot encoded phase indices)
                to be used for training and testing
        """
        x = self.x
        y = self.y
        testing_fraction = self.testing_fraction
        combined_xy = list(zip(x, y))
        shuffle(combined_xy)

        if testing_fraction == 0:
            train_x, train_y = zip(*combined_xy)
            test_x, test_y = None, None
            return np.array(train_x), np.array(train_y), test_x, test_y

        else:
            total_samples = len(combined_xy)
            n_testing = int(testing_fraction*total_samples)

            train_xy = combined_xy[n_testing:]
            train_x, train_y = zip(*train_xy)

            test_xy = combined_xy[:n_testing]
            test_x, test_y = zip(*test_xy)

            return np.array(train_x), np.array(train_y), np.array(test_x), np.array(test_y)

def train_model(x_train, y_train, n_phases, num_epochs, is_pdf, n_dense=[3100, 1200], dropout_rate=0.7):
    """
    Args:
        x_train: numpy array of simulated xrd spectra
        y_train: one-hot encoded vectors associated with reference phase indices
        n_phases: number of reference phases considered
        fmodel: filename to save trained model to
        n_dense: number of nodes comprising the two hidden layers in the neural network
        dropout_rate: fraction of connections excluded between the hidden layers during training
    Returns:
        model: trained and compiled tensorflow.keras.Model object
    """

    # Optimized architecture for PDF analysis
    if is_pdf:
        model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(filters=64, kernel_size=60, strides=1, padding='same', activation='relu'),
        tf.keras.layers.MaxPool1D(pool_size=3, strides=2, padding='same'),
        tf.keras.layers.MaxPool1D(pool_size=3, strides=2, padding='same'),
        tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='same'),
        tf.keras.layers.MaxPool1D(pool_size=1, strides=2, padding='same'),
        tf.keras.layers.MaxPool1D(pool_size=1, strides=2, padding='same'),
        tf.keras.layers.MaxPool1D(pool_size=1, strides=2, padding='same'),
        tf.keras.layers.Flatten(),
        CustomDropout(dropout_rate),
        tf.keras.layers.Dense(n_dense[0], activation='relu'),
        tf.keras.layers.BatchNormalization(),
        CustomDropout(dropout_rate),
        tf.keras.layers.Dense(n_dense[1], activation='relu'),
        tf.keras.layers.BatchNormalization(),
        CustomDropout(dropout_rate),
        tf.keras.layers.Dense(n_phases, activation='softmax')])

    # Optimized architecture for XRD analysis
    else:
        model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(filters=64, kernel_size=35, strides=1, padding='same', activation='relu'),
        tf.keras.layers.MaxPool1D(pool_size=3, strides=2, padding='same'),
        tf.keras.layers.Conv1D(filters=64, kernel_size=30, strides=1, padding='same', activation='relu'),
        tf.keras.layers.MaxPool1D(pool_size=3, strides=2, padding='same'),
        tf.keras.layers.Conv1D(filters=64, kernel_size=25, strides=1, padding='same', activation='relu'),
        tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='same'),
        tf.keras.layers.Conv1D(filters=64, kernel_size=20, strides=1, padding='same', activation='relu'),
        tf.keras.layers.MaxPool1D(pool_size=1, strides=2, padding='same'),
        tf.keras.layers.Conv1D(filters=64, kernel_size=15, strides=1, padding='same', activation='relu'),
        tf.keras.layers.MaxPool1D(pool_size=1, strides=2, padding='same'),
        tf.keras.layers.Conv1D(filters=64, kernel_size=10, strides=1, padding='same', activation='relu'),
        tf.keras.layers.MaxPool1D(pool_size=1, strides=2, padding='same'),
        tf.keras.layers.Flatten(),
        CustomDropout(dropout_rate),
        tf.keras.layers.Dense(n_dense[0], activation='relu'),
        tf.keras.layers.BatchNormalization(),
        CustomDropout(dropout_rate),
        tf.keras.layers.Dense(n_dense[1], activation='relu'),
        tf.keras.layers.BatchNormalization(),
        CustomDropout(dropout_rate),
        tf.keras.layers.Dense(n_phases, activation='softmax')])

    # Compile model
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False), optimizer=tf.keras.optimizers.Adam(), metrics=[tf.keras.metrics.CategoricalAccuracy()])

    # Fit model to training data
    model.fit(x_train, y_train, batch_size=32, epochs=num_epochs,
    validation_split=0.2, shuffle=True)

    return model

def test_model(model, test_x, test_y):
    """
    Args:
        model: trained tensorflow.keras.Model object
        x_test: feature matrix containing xrd spectra
        y_test: one-hot encoded vectors associated with
            the reference phases
    """
    _, acc = model.evaluate(test_x, test_y)
    print('Test Accuracy: ' + str(acc*100) + '%')

def main(xrd, num_epochs, testing_fraction, is_pdf, fmodel='Model.h5'):

    # Organize data
    obj = DataSetUp(xrd, testing_fraction)
    num_phases = obj.num_phases
    train_x, train_y, test_x, test_y = obj.split_training_testing()

    # Train model
    model = train_model(train_x, train_y, num_phases, num_epochs, is_pdf)

    # Save model
    model.save(fmodel, include_optimizer=False)

    # Test model is any data is reserved for testing
    if testing_fraction != 0:
        test_model(model, test_x, test_y)

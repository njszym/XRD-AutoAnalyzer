import numpy as np
import tensorflow as tf
from random import shuffle


def train_model(xrd, reserve_testing=False):
    """
    From a given set of X-ray diffraction spectra, train a convolutional
    neural network to perform phase identification.

    Args:
        xrd: a numpy array containing grouped xrd spectra.
            Each group is associated with a given reference phase.
            The shape of the array should be NxMx4501x1 where:
            N = number of reference phases,
            M = number of augmented spectra per reference phase
        reserve_testing: if True, train using only 80% of the given data.
            Otherwise, train on all spectra contained in xrd.
    Returns:
        None, saves the trained model as Model.h5 in the current directory.
    """

    ## Label each XRD spectrum with an associated one-hot vector (phase index)
    intensities, one_hot_vectors = [], []
    phase_indices = [val for val in range(len(xrd))]
    for (augmented_spectra, index) in zip(xrd, phase_indices):
        for pattern in augmented_spectra:
            intensities.append(pattern)
            assigned_vec = [[0]]*len(xrd)
            assigned_vec[index] = [1.0]
            one_hot_vectors.append(assigned_vec)
    intensities = np.array(intensities)
    one_hot_vectors = np.array(one_hot_vectors)

    ## Split into training/testing data if specified
    if reserve_testing:
        comb_data = list(zip(intensities, one_hot_vectors))
        shuffle(comb_data)
        total_samples = len(comb_data)
        train_test_split = int(0.8*total_samples)
        training_data = comb_data[:train_test_split]
        intensities, one_hot_vectors = zip(*training_data)
        test_data =	comb_data[train_test_split:]
        test_intensities, test_vecs = zip(*test_data)
        np.save('Test_Spectra', test_intensities)
        np.save('Test_Phases', test_vecs)

    # Define network structure
    model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(filters=64, kernel_size=35, strides=1, padding='same', activation = tf.nn.relu),
    tf.keras.layers.MaxPool1D(pool_size=3, strides=2, padding='same'),
    tf.keras.layers.Conv1D(filters=64, kernel_size=30, strides=1, padding='same', activation = tf.nn.relu),
    tf.keras.layers.MaxPool1D(pool_size=3, strides=2, padding='same'),
    tf.keras.layers.Conv1D(filters=64, kernel_size=25, strides=1, padding='same', activation = tf.nn.relu),
    tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='same'),
    tf.keras.layers.Conv1D(filters=64, kernel_size=20, strides=1, padding='same', activation = tf.nn.relu),
    tf.keras.layers.MaxPool1D(pool_size=1, strides=2, padding='same'),
    tf.keras.layers.Conv1D(filters=64, kernel_size=15, strides=1, padding='same', activation = tf.nn.relu),
    tf.keras.layers.MaxPool1D(pool_size=1, strides=2, padding='same'),
    tf.keras.layers.Conv1D(filters=64, kernel_size=10, strides=1, padding='same', activation = tf.nn.relu),
    tf.keras.layers.MaxPool1D(pool_size=1, strides=2, padding='same'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.7),
    tf.keras.layers.Dense(3100),
    tf.keras.layers.Dropout(0.7),
    tf.keras.layers.Dense(1200),
    tf.keras.layers.Dropout(0.7),
    tf.keras.layers.Dense(len(xrd))])

    # Compile model
    model.compile(loss=tf.nn.sigmoid_cross_entropy_with_logits, optimizer=tf.keras.optimizers.Adam(), metrics=[tf.keras.metrics.BinaryAccuracy()])

    # Fit model to training data
    model.fit(intensities, one_hot_vectors, batch_size=32, nb_epoch=2,
    validation_split=0.2, shuffle=True)

    # Evaluate trained accuracy
    _, acc = model.evaluate(int, phases)
    print('Trained Accuracy: ' + str(acc*100) + '%')

    model.save('Model.h5')


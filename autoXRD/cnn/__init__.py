import numpy as np
import tensorflow as tf
from random import shuffle


def train_and_save(reserve_testing=False):
    y_vals = np.load('XRD.npy')
    phases = [val for val in range(len(y_vals))]
    data = zip(y_vals, phases)

    binaries = []
    for pattern in data:
        for q in range(len(y_vals[0])):
            w = 1.0
            I = w*np.array(pattern[0][q])
            P = [pattern[1]]
            F = [w]
            binaries.append((I, P, F))

    classified_binaries = []
    for (I, P, F) in binaries: ## Encode one-hot vectors
        phase_zeroes = [[0.0]]*len(phases)
        for (phase_ind, phase_frac) in zip(P, F):
            if phase_frac == 1.0:
                phase_zeroes[phase_ind] = [1.0]
        mixture_comp = np.array(phase_zeroes).flatten()
        classified_binaries.append((I, mixture_comp))

    I = [pair[0] for pair in classified_binaries]
    mixture_class = [pair[1] for pair in classified_binaries]

    y_vals = np.array(I)
    phases = np.array(mixture_class)
    num_cat = len(phases[0]) ## no. of phases
    comb_data = list(zip(y_vals, phases))
    shuffle(comb_data)
    if reserve_testing:
        total_samples = len(comb_data)
        train_test_split = int(0.8*total_samples)
        training_data = comb_data[:train_test_split] ## 80% of simulated patterns used for training
        y_vals, phases = zip(*training_data)
        phases = np.array(phases)
        int = np.array(y_vals)
        test_data =	comb_data[train_test_split:] ## 20% reserved for testing
        y_test, phase_test = zip(*test_data)
        y_test = np.array(y_test)
        phase_test = np.array(phase_test)
        np.save('Test_Spectra', y_test)
        np.save('Test_Phases', phase_test)
    else:
        y_vals, phases = zip(*comb_data)

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
    tf.keras.layers.Dense(num_cat)])

    # Compile model
    model.compile(loss=tf.nn.sigmoid_cross_entropy_with_logits, optimizer=tf.keras.optimizers.Adam(), metrics=[tf.keras.metrics.BinaryAccuracy()])

    # Fit -- batch_size and nb_epoch subject to change
    model.fit(int, phases, batch_size=32, nb_epoch=2,
    validation_split=0.2, shuffle=True)

    # Evaluate
    _, acc = model.evaluate(int, phases)
    print('Trained Accuracy: ' + str(acc*100) + '%')

    model.save('Model.h5')


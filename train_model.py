"""
Title: Speaker Recognition Application with FFT and Conv1Dnet
Author: [Irfan Nafiz Shahan](https://www.github.com/irfannafiz)
Date created: 10/5/2023
Last modified: 8/8/2023
Hardware Used: GPU Nvidia RTX 3060 4GB
"""

import resampler
import slicer

"""
## Introduction

This example demonstrates how to create a model to classify speakers from the
frequency domain representation of speech recordings, obtained via Fast Fourier
Transform (FFT).

It shows the following:

- How to use `tf.data` to load, preprocess and feed audio streams into a model
- How to create a 1D convolutional network with residual
connections for audio classification.

Our process:

- We prepare a dataset of speech samples from different speakers, with the speaker as label.
- We add background noise to these samples to augment our data.
- We take the FFT of these samples.
- We train a 1D convnet to predict the correct speaker given a noisy FFT speech sample.

Note:

- This example should be run with TensorFlow 2.3 or higher, or `tf-nightly`.
- The noise samples in the dataset need to be resampled to a sampling rate of 16000 Hz
before using the code in this example. In order to do this, you will need to have
installed `ffmpg`.
"""

"""
## Setup
"""

import os
import shutil
import numpy as np

import tensorflow as tf
from tensorflow import keras

from pathlib import Path
from IPython.display import display, Audio

# to save the 'noises' variable to be used for later on in the application
import joblib


# and save it to the 'Downloads' folder in your HOME directory
DATASET_ROOT = "data/custom"

# The folders in which we will put the audio samples and the noise samples
AUDIO_SUBFOLDER = "audio"
NOISE_SUBFOLDER = "noise"

DATASET_AUDIO_PATH = os.path.join(DATASET_ROOT, AUDIO_SUBFOLDER)
DATASET_NOISE_PATH = os.path.join(DATASET_ROOT, NOISE_SUBFOLDER)

# Percentage of samples to use for validation
VALID_SPLIT = 0.2
SHUFFLE_SEED = 43
SAMPLING_RATE = 16000

# The factor to multiply the noise with according to:
#   noisy_sample = sample + noise * prop * scale
#      where prop = sample_amplitude / noise_amplitude
SCALE = 0.5

BATCH_SIZE = 4
EPOCHS = 100


"""
## Dataset generation
"""


def paths_and_labels_to_dataset(audio_paths, labels):
    """Constructs a dataset of audios and labels."""
    path_ds = tf.data.Dataset.from_tensor_slices(audio_paths)
    audio_ds = path_ds.map(
        lambda x: path_to_audio(x), num_parallel_calls=tf.data.AUTOTUNE
    )
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    return tf.data.Dataset.zip((audio_ds, label_ds))


def path_to_audio(path):
    """Reads and decodes an audio file."""
    audio = tf.io.read_file(path)
    audio, _ = tf.audio.decode_wav(audio, 1, SAMPLING_RATE)
    return audio


def add_noise(audio, noises=None, scale=0.5):
    if noises is not None:
        # Create a random tensor of the same size as audio ranging from
        # 0 to the number of noise stream samples that we have.
        tf_rnd = tf.random.uniform(
            (tf.shape(audio)[0],), 0, noises.shape[0], dtype=tf.int32
        )
        noise = tf.gather(noises, tf_rnd, axis=0)

        # Get the amplitude proportion between the audio and the noise
        prop = tf.math.reduce_max(audio, axis=1) / tf.math.reduce_max(noise, axis=1)
        prop = tf.repeat(tf.expand_dims(prop, axis=1), tf.shape(audio)[1], axis=1)

        # Adding the rescaled noise to audio
        audio = audio + noise * prop * scale

    return audio


def audio_to_fft(audio):
    # Since tf.signal.fft applies FFT on the innermost dimension,
    # we need to squeeze the dimensions and then expand them again
    # after FFT
    audio = tf.squeeze(audio, axis=-1)
    fft = tf.signal.fft(
        tf.cast(tf.complex(real=audio, imag=tf.zeros_like(audio)), tf.complex64)
    )
    fft = tf.expand_dims(fft, axis=-1)

    # Return the absolute value of the first half of the FFT
    # which represents the positive frequencies
    return tf.math.abs(fft[:, : (audio.shape[1] // 2), :])

if __name__ == '__main__':
    """
    # Data preparation

    - An `audio` folder which will contain all the per-speaker speech sample folders
    - A `noise` folder which will contain all the noise samples
    """

    """
    ## Sorting the dataset within Custom folder
    """

    # If folder `audio`, does not exist, create it, otherwise do nothing
    if os.path.exists(DATASET_AUDIO_PATH) is False:
        os.makedirs(DATASET_AUDIO_PATH)

    # If folder `noise`, does not exist, create it, otherwise do nothing
    if os.path.exists(DATASET_NOISE_PATH) is False:
        os.makedirs(DATASET_NOISE_PATH)

    for folder in os.listdir(DATASET_ROOT):
        if os.path.isdir(os.path.join(DATASET_ROOT, folder)):
            if folder in [AUDIO_SUBFOLDER, NOISE_SUBFOLDER]:
                # If folder is `audio` or `noise`, do nothing
                continue
            elif folder in ["other", "_background_noise_"]:
                # If folder is one of the folders that contains noise samples,
                # move it to the `noise` folder
                shutil.move(
                    os.path.join(DATASET_ROOT, folder),
                    os.path.join(DATASET_NOISE_PATH, folder),
                )
            else:
                # Otherwise, it should be a speaker folder, then move it to
                # `audio` folder
                shutil.move(
                    os.path.join(DATASET_ROOT, folder),
                    os.path.join(DATASET_AUDIO_PATH, folder),
                )

    """
    # Noise preparation

    In this section:

    - We load all noise samples (which should have been resampled to 16000)
    - We split those noise samples to chunks of 16000 samples which
    correspond to 1 second duration each
    """

    # Get the list of all noise files
    noise_files = []
    for subdir in os.listdir(DATASET_NOISE_PATH):
        subdir_path = Path(DATASET_NOISE_PATH) / subdir
        if os.path.isdir(subdir_path):
            noise_files += [
                os.path.join(subdir_path, filepath)
                for filepath in os.listdir(subdir_path)
                if filepath.endswith(".wav")
            ]

    print(
        "Found {} files belonging to {} directories".format(
            len(noise_files), len(os.listdir(DATASET_NOISE_PATH))
        )
    )

    """
    ## Resample all noise samples to 16000 Hz
    """
    resampleNoiseToggle = 1
    if resampleNoiseToggle == 1:
        resampler.resample(DATASET_NOISE_PATH)

    noises = []
    for path in noise_files:
        sample = slicer.load_noise_and_slice(path)
        if sample:
            noises.extend(sample)
    noises = tf.stack(noises)

    # Save the noise samples to disk for further use
    joblib.dump(noises, 'saved_variable.joblib')

    print(
        "{} noise files were split into {} noise samples where each is {} sec. long".format(
            len(noise_files), noises.shape[0], noises.shape[1] // SAMPLING_RATE
        )
    )

    # Get the list of audio file paths along with their corresponding labels
    class_names = os.listdir(DATASET_AUDIO_PATH)

    # if desktop.ini is present remove desktop.ini file from class_names
    if 'desktop.ini' in class_names:
        class_names.remove('desktop.ini')

    print("Our class names: {}".format(class_names,))

    audio_paths = [] # nested list of audio paths
    labels = [] # nested list of labels which are indexes of the class_names list for each slice of audio
    for label, name in enumerate(class_names):
        print("Processing speaker {}".format(name,))

        dir_path = Path(DATASET_AUDIO_PATH) / name
        speaker_sample_paths = [os.path.join(dir_path, filepath) for filepath in os.listdir(dir_path) if filepath.endswith(".wav")]
        audio_paths += speaker_sample_paths
        labels += [label] * len(speaker_sample_paths)

    print(
        "Found {} files belonging to {} classes.".format(len(audio_paths), len(class_names))
    )

    # Shuffle
    rng = np.random.RandomState(SHUFFLE_SEED)
    rng.shuffle(audio_paths)
    rng.shuffle(audio_paths)
    rng = np.random.RandomState(SHUFFLE_SEED)
    rng.shuffle(labels)
    rng.shuffle(labels)

    # Split into training and validation 80:20
    num_val_samples = int(VALID_SPLIT * len(audio_paths))
    print("Using {} files for training.".format(len(audio_paths) - num_val_samples))
    train_audio_paths = audio_paths[:-num_val_samples]
    train_labels = labels[:-num_val_samples]

    print("Using {} files for validation.".format(num_val_samples))
    valid_audio_paths = audio_paths[-num_val_samples:]
    valid_labels = labels[-num_val_samples:]

    # Create 2 datasets, one for training and the other for validation
    train_ds = paths_and_labels_to_dataset(train_audio_paths, train_labels)
    train_ds = train_ds.shuffle(buffer_size=BATCH_SIZE * 8, seed=SHUFFLE_SEED).batch(
        BATCH_SIZE
    )

    valid_ds = paths_and_labels_to_dataset(valid_audio_paths, valid_labels)
    valid_ds = valid_ds.shuffle(buffer_size=BATCH_SIZE * 8, seed=SHUFFLE_SEED).batch(BATCH_SIZE) # batch size is 32 for validation set


    # Add noise to the training set
    train_ds = train_ds.map(
        lambda x, y: (add_noise(x, noises, scale=SCALE), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    # Transform audio wave to the frequency domain using `audio_to_fft`
    train_ds = train_ds.map(
        lambda x, y: (audio_to_fft(x), y), num_parallel_calls=tf.data.AUTOTUNE
    )
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

    valid_ds = valid_ds.map(
        lambda x, y: (audio_to_fft(x), y), num_parallel_calls=tf.data.AUTOTUNE
    )
    valid_ds = valid_ds.prefetch(tf.data.AUTOTUNE)

    """
    ## Model Definition
    """


    def residual_block(x, filters, conv_num=3, activation="relu"):
        # Shortcut
        s = keras.layers.Conv1D(filters, 1, padding="same")(x)
        for i in range(conv_num - 1):
            x = keras.layers.Conv1D(filters, 3, padding="same")(x)
            x = keras.layers.Activation(activation)(x)
        x = keras.layers.Conv1D(filters, 3, padding="same")(x)
        x = keras.layers.Add()([x, s])
        x = keras.layers.Activation(activation)(x)
        return keras.layers.MaxPool1D(pool_size=2, strides=2)(x)


    def build_model(input_shape, num_classes):
        inputs = keras.layers.Input(shape=input_shape, name="input")

        x = residual_block(inputs, 16, 2)
        x = residual_block(x, 32, 2)
        x = residual_block(x, 64, 3)
        x = residual_block(x, 128, 3)
        x = residual_block(x, 128, 3)

        x = keras.layers.AveragePooling1D(pool_size=3, strides=3)(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(256, activation="relu")(x)
        x = keras.layers.Dense(128, activation="relu")(x)

        outputs = keras.layers.Dense(num_classes, activation="softmax", name="output")(x)

        return keras.models.Model(inputs=inputs, outputs=outputs)


    model = build_model((SAMPLING_RATE // 2, 1), len(class_names))

    print("\n _________MODEL SUMMARY__________:")
    model.summary()

    # Compile the model using Adam's default learning rate
    model.compile(
        optimizer="Adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    # Add callbacks:
    # 'EarlyStopping' to stop training when the model is not enhancing anymore
    # 'ModelCheckPoint' to always keep the model that has the best val_accuracy
    model_save_filename = "model.h5"

    earlystopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    mdlcheckpoint_cb = keras.callbacks.ModelCheckpoint(model_save_filename, verbose=1,
                                                       monitor="val_accuracy", save_best_only=True)

    """
    ## Training
    """

    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=valid_ds,
        callbacks=[earlystopping_cb, mdlcheckpoint_cb],
    )

    """
    ## Evaluation
    """

    print("/n _________MODEL EVALUATION__________:")
    print(model.evaluate(valid_ds))

    """
    ## Demonstration
    
    Let's take some samples and:
    
    - Predict the speaker
    - Compare the prediction with the real speaker
    - Listen to the audio to see that despite the samples being noisy,
    the model is still pretty accurate
    """

    SAMPLES_TO_DISPLAY = 20

    test_ds = paths_and_labels_to_dataset(valid_audio_paths, valid_labels)
    test_ds = test_ds.shuffle(buffer_size=BATCH_SIZE * 8, seed=SHUFFLE_SEED).batch(BATCH_SIZE)

    test_ds = test_ds.map(
        lambda x, y: (add_noise(x, noises, scale=SCALE), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    for audios, labels in test_ds.take(1):
        # Get the signal FFT
        ffts = audio_to_fft(audios)
        # Predict
        y_pred = model.predict(ffts)
        # Take random samples
        rnd = np.random.randint(0, BATCH_SIZE, SAMPLES_TO_DISPLAY)
        audios = audios.numpy()[rnd, :, :]
        labels = labels.numpy()[rnd]
        y_pred = np.argmax(y_pred, axis=-1)[rnd]

        for index in range(SAMPLES_TO_DISPLAY):
            # For every sample, print the true and predicted label
            # as well as run the voice with the noise
            print(
                "Speaker:\33{} {}\33[0m\tPredicted:\33{} {}\33[0m".format(
                    "[92m" if labels[index] == y_pred[index] else "[91m",
                    class_names[labels[index]],
                    "[92m" if labels[index] == y_pred[index] else "[91m",
                    class_names[y_pred[index]],
                )
            )
            display(Audio(audios[index, :, :].squeeze(), rate=SAMPLING_RATE))
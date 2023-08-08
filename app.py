# importing libraries
import os
import shutil
import sys

import joblib
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import speech_recognition as sr
import tensorflow as tf
from tensorflow.python.keras.models import load_model

import train_model

DATASET_ROOT = "data/custom"

# The folders in which we will put the audio samples and the noise samples
AUDIO_SUBFOLDER = "audio"
NOISE_SUBFOLDER = "noise"
TEMP_SUBFOLDER = "temp/temp_record"

DATASET_AUDIO_PATH = os.path.join(DATASET_ROOT, AUDIO_SUBFOLDER)
DATASET_NOISE_PATH = os.path.join(DATASET_ROOT, NOISE_SUBFOLDER)
DATASET_TEMP_PATH = os.path.join(DATASET_ROOT, TEMP_SUBFOLDER)

def record_audio():
    """
    Records a 10 second clip using sr.
    """
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Recording...")
        try:
            audio = r.listen(source, timeout=10)
            print("Recording complete.")
        except sr.WaitTimeoutError:
            print("No audio detected. Recording is silent.")
            return None  # Return None or any other appropriate value to indicate silent recording

    return audio

def save_audio(audio_record, filename='temp_record'):
    """
    Save audio as a WAV file in the specified path.

    :param audio_record: Audio data from the Python SpeechRecognition library.
    :param filename: Filename to save the audio (default: 'temp_record.wav').
    """
    file_path = os.path.join(DATASET_ROOT, filename + '.wav')
    with open(file_path, "wb") as file:
        file.write(audio_record.get_wav_data())
    print("Audio saved as", file_path)


# remove temp_record.wav file and temp folder
def delete_temp():
    if os.path.exists(os.path.join(DATASET_ROOT, 'temp_record.wav')):
        os.remove(os.path.join(DATASET_ROOT, 'temp_record.wav'))
    if os.path.exists(os.path.join(DATASET_ROOT, 'temp')):
        shutil.rmtree(os.path.join(DATASET_ROOT, 'temp'))
    print("Temp files deleted successfully!")


# set variable for if you want to record a custom audio or use a preexisting audio in the root folder named "temp_record.wav"
setCustomRecord = True  # by default set to true, if you want to use a preexisting audio set to false
def record_protocol():  # when record is pressed this will happen
    if setCustomRecord:
        delete_temp()
        print("Please say the pass phrase 'Hello D S P 1 2 3 4 5'")
        wait_for_enter()

        audio = record_audio()
        while audio == None:
            audio = record_audio()

        print("Saving audio temporarily!")
        save_audio(audio) # temp save
        print("Audio saved successfully!")

    # slice and save temp audio in temp folder
    script_path = 'audio_slicer.py'
    # Run the script using os.system()
    os.system(f'python {script_path}')
    print("Audio sliced, preprocessed and saved successfully!")


def get_maximum_occurence(class_names):
    # Count the occurrences of each string
    name_counts = Counter(class_names)

    # Find the maximum occurrence count
    max_count = max(name_counts.values())

    # Filter strings that occur with the maximum count
    most_common_names = [name for name, count in name_counts.items() if count == max_count]

    return most_common_names

# get temp slices paths
def get_temp_slices_paths():
    temp_paths = []
    temp_slice_labels = []
    temp_slice_paths = [os.path.join(DATASET_TEMP_PATH, filepath) for filepath in os.listdir(DATASET_TEMP_PATH) if filepath.endswith(".wav")]
    temp_paths += temp_slice_paths
    temp_slice_labels += [0] * len(temp_slice_paths) # redundant for temp files hence 0s
    print(
        "Found {} files in temp folder.".format(len(temp_slice_paths))
    )
    return temp_paths, temp_slice_labels


def load_necessary_files():
    noises = joblib.load('saved_variable.joblib')


    class_names = os.listdir(DATASET_AUDIO_PATH)
    # if desktop.ini is present remove desktop.ini file from class_names
    if 'desktop.ini' in class_names:
        class_names.remove('desktop.ini')

    # load model
    model = load_model("model.h5")

    return noises, class_names, model

# main part of the application: will run until the user exits the program
def record_and_predict(debug=False):
    record_protocol()

    temp_paths, temp_labels = get_temp_slices_paths()
    temp_ds = train_model.paths_and_labels_to_dataset(temp_paths, temp_labels)
    temp_ds = temp_ds.shuffle(buffer_size=train_model.BATCH_SIZE * 8, seed=train_model.SHUFFLE_SEED).batch(train_model.BATCH_SIZE)

    temp_ds = temp_ds.map(
        lambda x, y: (train_model.add_noise(x, noises, scale=train_model.SCALE), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    prediction = []
    for audios, labels in temp_ds.take(1):
        # Get the signal FFT
        ffts = train_model.audio_to_fft(audios)

        # Predict
        y_pred_raw = model.predict(ffts)
        # Get probability of predictions that are measured as the maximum
        y_pred_probs = np.max(y_pred_raw, axis=-1)
        # get index of maximum probability
        y_pred = np.argmax(y_pred_raw, axis=-1)

        # debug prints
        if debug:
            print("DEBUG PRINTS:"
                  "______________________________")
            print(class_names)
            print("Raw predictions probabilities: \n", y_pred_raw)
            print("After sorting prediction probabilities: ", y_pred_probs)
            print("Corresponding probability percentages: ", np.array(y_pred_probs)*100)
            print("Index of prediction using argmax: ", y_pred)
            print("______________________________")

        for index in range(len(y_pred)):
            # For every sample, print the true and predicted label
            # as well as run the voice with the noise
            prediction.append(class_names[y_pred[index]])
            print("Predicted: \33[93m{}\33[0m".format(class_names[y_pred[index]]))
            print("Probability: \33[93m{}\33[0m".format(y_pred_probs[index]))

    max_pred = get_maximum_occurence(prediction)
    if debug:
        print("Maximum occurence: ", max_pred)

    if len(max_pred) == 1:
        print("\nFinal prediction is that the speaker may be \33[92m{}\33[0m".format(str(max_pred)))
        print("Probability of prediction is \33[92m{}\33[0m".format(str(np.average(y_pred_probs)*100) + "%"))
        return max_pred[0]
    elif len(max_pred) > 2:
        print("\nFinal prediction is that the speaker may be one of \33[91m{}\33[0m".format(str(max_pred)))
        print("It is not possible to determine the speaker with high confidence.")
        print("Please try again!")
    else:
        print("\nFinal prediction is that the speaker may be either of \33[93m{}\33[0m".format(str(max_pred)))
        print("It is not possible to determine the speaker with high confidence.")
        print("Please try again!")

    return max_pred


def is_valid_filename(filename):
    """
    Check if a string is valid to be used as a filename.

    Returns True if the filename is valid, False otherwise.
    """
    # reject empty filename
    if not filename:
        return False

    # any invalid characters
    invalid_chars = r'<>:"/\|?*'
    if any(char in invalid_chars for char in filename):
        return False

    # reserved name in the operating system
    reserved_names = ["CON", "PRN", "AUX", "NUL", "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9", "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9"]
    if filename.upper() in reserved_names:
        return False

    # filename is a valid path
    try:
        # Attempt to join the filename with a dummy path
        # If it raises an exception, the filename is invalid
        _ = os.path.join(DATASET_ROOT, filename)
    except (ValueError, OSError):
        return False

    # If all checks passed, the filename is valid
    return True


def prompt_name_and_save(audio):
    while True:
        person_name = input("What is your name? >>")
        filename = person_name.replace(" ", "_")
        if is_valid_filename(filename + ".wav"):
            save_audio(audio, filename)
            print("Your name is saved as " + str(filename) + " and recorded audio is saved as " + str(filename) + ".wav! in the dataset folder.")
            break
        else:
            print("Invalid person name. Use valid characters only.")


def wait_for_enter():
    input("Press ENTER to continue.")


def retraining_protocol():
    # retrain the model by running train_model.py
    script_path = 'train_model.py'
    os.system(f'python {script_path}')


def preprocessing_protocol():
    # slice and save any new audio in audio folder
    script_path = 'audio_slicer.py'
    os.system(f'python {script_path}')
    print("Audio sliced, preprocessed and saved successfully!")


def check_audio_for_validity(audio, for_dataset=False):
    audio_length = len(audio.frame_data) / train_model.SAMPLING_RATE
    print("Audio length is {} seconds.".format(audio_length))
    if audio_length is None:
        print("No audio was recorded. Please try again.")
        wait_for_enter()
        return False
    if audio_length < train_model.BATCH_SIZE:
        print("Audio length is too short for running prediction (min 4 sec). Please try again.")
        wait_for_enter()
        return False
    if for_dataset and audio_length < train_model.BATCH_SIZE*10:
        print("Audio length is too short for a dataset entry (min 40 sec). Please try again.")
        wait_for_enter()
        return False
    return True


def end_credits():
    print("\n _______________________________________"
          "\nThank you for using this speaker recognition application!\n"
          "\n Made by group 12:"
          "\n Registration No.: 2019338055, 49, 38, 73"
          "\nThis application was developed as a part of our undergraduate "
          "\nproject in Digital Signal Processing Lab Course EEE332."
          "\n"
          "\nIf you have any feedback, please email me at: irfannafizislive@gmail.com"
          "\n"
          "\nIrfan Nafiz Shahan,"
          "\nDepartment of Electrical and Electronic Engineering, "
          "\nShahjalal University of Science and Technology, "
          "\nSylhet, Bangladesh.")


def import_arguments(d):
    if len(sys.argv) != 2:
        print("For debug mode use -d as argument. Example: python app.py -d")

    if len(sys.argv) == 2:
        debug = sys.argv[1]
        if debug.lower() == '-d':
            d = True
            print("DEBUG MODE")

        else:
            print("For debug mode use -d as argument. Example: python app.py -d")

    return d


ApplicationRunning = True
debug = False
if __name__ == '__main__':

    debug = import_arguments(debug)
    while ApplicationRunning:
        noises, class_names, model = load_necessary_files()
        username = record_and_predict(debug)

        if type(username) == str:
            username = [username]

        if len(username) > 1:
            print(f"Multiple users detected: {len(username)} users. "
                  "Please try again.")
            continue
        # ask if the prediction is correct
        user_input = input("Is the prediction correct? Y/N\n>>").upper()
        while user_input not in ["Y", "N"]:
            user_input = input("Invalid input. Please enter Y or N.\n>>").upper()

        if user_input == "Y":
            print(f"Welcome back {username[0]}!")

        # if no then ask to record a passphrase dataset
        else:
            print("Please record a new dataset, saying 'Hello D S P 1 2 3 4 5' about 10 times. "
                  "\nEnsure you dont have any background noise, and make sure you pronounce "
                  "\nthem normally as you would.")
            wait_for_enter()

            audio = record_audio()
            while audio == None:
                audio = record_audio()

            prompt_name_and_save(audio)

            # slice and save any new audio in audio folder
            preprocessing_protocol()

            # retrain the model
            retraining_protocol()

        # ask if the user wants to continue
        user_input = input("Do you want to try recording again? Y/N").upper()
        while user_input not in ["Y", "N"]:
            user_input = input("Invalid input. Please enter Y or N.\n>>").upper()
        if user_input == "N":
            ApplicationRunning = False
            end_credits()
            break


# def predict_speaker_ml(audio):
#     elif model == "ml":
#         guess = predict_speaker_ml(audio)  # TODO: make this predict_speaker_ml function, return false if no guess
#
#     if guess: # TODO: Show these prints in a display box
#         print("Hello, is this ___?")
#         cmd = input("Y/N>>") # TODO: Make this a button
#         while cmd not in ["Y", "N"]:
#             if cmd == "Y":
#                 print("Hey there ____!")
#                 break
#             # if guess not correct then prompt the user for name
#             else:
#                 print("Oh I don't recognize you, can you please provide your name?")
#                 # TODO: Open a input name box here
#                 prompt_name(audio)
#                 break
#
#     # if no guess valid then just prompt directly
#     else:
#         prompt_name(audio)




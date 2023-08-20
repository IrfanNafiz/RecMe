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


def percent_complete(percentage):
    print(f"Total complete: {percentage}%")


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

# TODO setCustomRecord situation is not created yet
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

        # create waveform and fft plot for ui output
        run_script('generate_plot.py')

    # slice and save temp audio in temp folder
    run_script('audio_slicer.py')
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
    speaker_confidence_scores = {}  # Store confidence scores for each speaker
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

            # Store confidence scores for each speaker
        for index in range(len(y_pred)):
            speaker_confidence_scores.setdefault(class_names[y_pred[index]], []).append(y_pred_probs[index])

        for index in range(len(y_pred)):
            # For every sample, print the true and predicted label
            # as well as run the voice with the noise
            prediction.append(class_names[y_pred[index]])
            print("Predicted: \33[93m{}\33[0m".format(class_names[y_pred[index]]))
            print("Probability: \33[93m{}\33[0m".format(y_pred_probs[index]))

    max_pred = get_maximum_occurence(prediction)
    if debug:
        print("Maximum occurence: ", max_pred)

    # TODO make sure the most predicted speaker's average probability is given, and not the highest probability of all speakers
    # Calculate weighted probabilities for each speaker
    weighted_speaker_probs = {}
    for speaker, conf_scores in speaker_confidence_scores.items():
        average_conf_score = np.mean(conf_scores)  # Calculate the average confidence score
        weighted_speaker_probs[speaker] = average_conf_score

    # Get the most highly predicted speaker
    most_highly_predicted_speaker = max(weighted_speaker_probs, key=weighted_speaker_probs.get)
    probability_of_most_highly_predicted_speaker = weighted_speaker_probs[most_highly_predicted_speaker]

    # print("Most Highly Predicted Speaker:", most_highly_predicted_speaker)
    # print("Probability of Most Highly Predicted Speaker:", probability_of_most_highly_predicted_speaker)


    if len(max_pred) == 1:
        print("\nFinal prediction is that the speaker may be \33[92m{}\33[0m".format(most_highly_predicted_speaker))
        print("Probability of prediction is \33[92m{}\33[0m".format(str(probability_of_most_highly_predicted_speaker*100) + "%"))
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
    input("Press ENTER/CMD to continue.")


def run_script(script_path):
    os.system(f'python {script_path}')


def retraining_protocol():
    # retrain the model by running train_model.py
    print("Retraining the model...")
    run_script('train_model.py')


def preprocessing_protocol():
    # slice and save any new audio in audio folder
    delete_temp()
    run_script('audio_slicer.py')
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


def add_person():
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

def continue_or_exit():
    # ask if the user wants to continue
    user_input = input("Do you want to try recording again? Y/N").upper()
    while user_input not in ["Y", "N"]:
        user_input = input("Invalid input. Please enter Y or N.\n>>").upper()
    if user_input == "N":
        end_credits()
        return False
    else:
        return True

def import_arguments(d, ui_mode):
    if len(sys.argv) != 2:
        print("For debug mode use -d as an argument. Example: python app.py -d")

    if len(sys.argv) == 2:
        arg = sys.argv[1]

        if arg.lower() == '-d':
            d = True
            print("DEBUG MODE")
        elif arg.lower() == '-ui':
            ui_mode = True
            # print("UI MODE")
        else:
            print("For debug mode use -d as an argument. Example: python app.py -d")

    return d, ui_mode

# application-specific arguments and variables
ApplicationRunning = True
debug = False
ui_mode = False

if __name__ == '__main__':

    debug, ui_mode = import_arguments(debug, ui_mode)
    percent_complete(5)

    while ApplicationRunning:

        noises, class_names, model = load_necessary_files()
        username = record_and_predict(debug)

        if type(username) == str:
            username = [username]

        if len(username) > 1:
            user_input = input("Multiple users detected. Do you want to add a new user? Y/N\n>>").upper()
            while user_input not in ["Y", "N"]:
                user_input = input("Invalid input. Please enter Y or N.\n>>").upper()
            if user_input == "Y":
                add_person()
                ApplicationRunning = continue_or_exit()
                if ApplicationRunning == False:
                    break
                else:
                    continue

        # ask if the prediction is correct
        else:
            user_input = input("Is the prediction correct? Y/N\n>>").upper()
            while user_input not in ["Y", "N"]:
                user_input = input("Invalid input. Please enter Y or N.\n>>").upper()

            if user_input == "Y":
                print(f"Welcome back {username[0]}!")

            # if no then ask to record a passphrase dataset
            else:
                add_person()

        # ask if the user wants to continue
        ApplicationRunning = continue_or_exit()

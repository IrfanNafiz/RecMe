# importing libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import speech_recognition as sr
import tensorflow as tf

DATASET_ROOT = "data/custom"

# The folders in which we will put the audio samples and the noise samples
AUDIO_SUBFOLDER = "audio"
NOISE_SUBFOLDER = "noise"

DATASET_AUDIO_PATH = os.path.join(DATASET_ROOT, AUDIO_SUBFOLDER)
DATASET_NOISE_PATH = os.path.join(DATASET_ROOT, NOISE_SUBFOLDER)


# TODO: Make flow chart of the app, include gui
# Current flow: ask for voice, save voice as temp file, check temp file with existing saved recordings with names, try a close possibility, prompts is this the user, if yes then okay
# if not the user, ask for name, save temp file as person name wav file in records, make model save as well, delete temp file from previously
# compare temp_audio with existing records, if matched, spew name, or try, then ask if correct, otherwise prompt for new perons name


def record_protocol(model="naive"):  # when record is pressed this will happen
    audio = record_audio()
    # save_audio(audio) # temp save
    if model == "naive":
        guess = predict_speaker_naive(audio)  # TODO: make this predict_speaker_naive function, return false if no guess
    elif model == "ml":
        guess = predict_speaker_ml(audio)  # TODO: make this predict_speaker_ml function, return false if no guess

    if guess: # TODO: Show these prints in a display box
        print("Hello, is this ___?")
        cmd = input("Y/N>>") # TODO: Make this a button
        while cmd not in ["Y", "N"]:
            if cmd == "Y":
                print("Hey there ____!")
                break
            # if guess not correct then prompt the user for name
            else:
                print("Oh I don't recognize you, can you please provide your name?")
                # TODO: Open a input name box here
                prompt_name(audio)
                break

    # if no guess valid then just prompt directly
    else:
        prompt_name(audio)



def record_audio():
    """
    Records a 5 second clip.
    """
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Recording...")
        audio = r.listen(source, timeout=5)
    return audio

def save_audio(audio, filename='temp_record', path='/recordings/'):
    """
    Save audio as a WAV file in the specified path.

    :param audio: Audio data from the Python SpeechRecognition library.
    :param filename: Filename to save the audio (default: 'temp_record.wav').
    :param path: Directory path where the audio file will be saved (default: current directory).
    """
    file_path = os.path.join(path, filename + '.wav')
    with open(file_path, "wb") as file:
        file.write(audio.get_wav_data())
    print("Audio saved as", file_path)


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
        _ = os.path.join('dummy_path', filename)
    except (ValueError, OSError):
        return False

    # If all checks passed, the filename is valid
    return True


def prompt_name(audio):
    while True:
        person_name = input("What is your name? >>")
        if is_valid_filename(person_name):
            print("Your name is saved as " + str(person_name) + " and recorded audio is saved as " + str(person_name) + ".wav!.")
            save_audio(audio, person_name)
            break
        else:
            print("Invalid person name. Use valid characters only.")


def delete_file(file_path):
    """
    Delete a specific file.

    :param file_path: Path to the file to be deleted.
    """
    try:
        os.remove(file_path)
        print("File deleted:", file_path)
    except OSError as e:
        print(f"Error deleting the file: {e}")






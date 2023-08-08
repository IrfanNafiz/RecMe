"""This python script will take the raw audio files kept in the main folder, convert them to mono resample them each to 16khz, then slice
them into 1 second chunks and save them in the same folder. The script will also create a folder for each audio file
named after the audio file. This is done to make it easier to train the model later on."""

import soundfile as sf
import librosa
import os

DATASET_ROOT = "data\custom"
DATASET_AUDIO_ROOT = os.path.join(DATASET_ROOT, "audio")
DATASET_TEMP_ROOT = os.path.join(DATASET_ROOT, "temp")

def slice_and_save_audio(input_file, output_folder):
    """Slice the input file into 1 second slices and save them in the output folder.
    Resanple to 16 kHz. And also convert to mono if needed."""

    sample_rate = 16000
    slice_length = sample_rate  # 1 second slices

    audio, sr = librosa.load(input_file)

    # Ensure mono channel
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    # Resample to 16 kHz
    if sr != sample_rate:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)

    num_slices = len(audio) // slice_length

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for i in range(num_slices):
        slice_audio = audio[i * slice_length : (i + 1) * slice_length]
        output_path = os.path.join(output_folder, f"{os.path.splitext(file)[0]}_{i + 1}.wav")
        sf.write(output_path, slice_audio, sample_rate)


if __name__ == "__main__":
    files_in_root_directory = os.listdir(DATASET_ROOT)
    files_in_audio_directory = os.listdir(DATASET_AUDIO_ROOT)
    print("Files in root directory: ", files_in_root_directory)
    print("Files in audio directory: ", files_in_audio_directory)

    # remove desktop.ini from audio dir list
    if 'desktop.ini' in files_in_audio_directory:
        files_in_audio_directory.remove('desktop.ini')

    # remove desktop.ini from root dir list
    if 'desktop.ini' in files_in_root_directory:
        files_in_root_directory.remove('desktop.ini')

    # slice and save audio if in root directory but not in audio directory
    for file in files_in_root_directory:
        # check if a temp file was created by app.py
        if file == "temp_record.wav":
            print(f"Found temporary file {file}. Slicing and saving...")
            input_file = os.path.join(DATASET_ROOT, file)
            output_folder = os.path.join(DATASET_TEMP_ROOT, os.path.splitext(file)[0])
            slice_and_save_audio(input_file, output_folder)
            print(f"Sliced and saved temporary file {file} into {output_folder}")
            continue

        # check if file is a wav file and if it is not already processed in the audio directory
        elif file.endswith(".wav") and os.path.splitext(file)[0] not in files_in_audio_directory:
            input_file = os.path.join(DATASET_ROOT, file)
            output_folder = os.path.join(DATASET_AUDIO_ROOT, os.path.splitext(file)[0])
            slice_and_save_audio(input_file, output_folder)
            print(f"Sliced and saved {file} into {output_folder}")

        if file.endswith(".wav") and os.path.splitext(file)[0] in files_in_audio_directory and os.path.splitext(file)[0] != "temp_record":
            print(f"{file} is already processed. Skipping...")
            continue

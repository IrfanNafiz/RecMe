import PySimpleGUI as sg
import sounddevice as sd
import numpy as np

# Constants
SAMPLE_RATE = 44100  # Sample rate for audio recording
RECORD_SECONDS = 5  # Duration of audio recording

# Dummy function for speaker identification
def identify_speaker(audio_data):
    # Placeholder implementation
    # Replace with your actual speaker identification logic
    return "John Doe"

# Create the GUI layout
layout = [
    [sg.Text("RecMe - The Speaker Identifier", font=("Helvetica", 16))],
    [sg.Text("", key="-INFO-", size=(40, 1))],
    # [sg.Graph(canvas_size=(500, 200), graph_bottom_left=(0, -1), graph_top_right=(RECORD_SECONDS, 1), key="-WAVEFORM-")],
    [sg.Button(image_filename="record_icon.png", image_subsample=2, border_width=0, key="-RECORD-", tooltip="Record", button_color=("white", "red"))],
    [sg.Text("Is this speaker X speaking?")],
    [sg.Button("Yes", key="-YES-", size=(10, 1)), sg.Button("No", key="-NO-", size=(10, 1))],
    [sg.Text("What is your name?"), sg.InputText(key="-NAME-")],
    [sg.Button("Submit", key="-SUBMIT-", size=(10, 1))]
]

# Create the window
window = sg.Window("RecMe", layout, finalize=True)

# Set up audio recording
recording = False
audio_buffer = np.array([])

# Audio callback function for live audio display
def audio_callback(indata, frames, time, status):
    global audio_buffer
    if recording:
        audio_buffer = np.append(audio_buffer, indata[:, 0])

# Main event loop
while True:
    event, values = window.read(timeout=100) # window automatically closes after 100 seconds TODO: remove in release

    # Exit if the window is closed
    if event == sg.WINDOW_CLOSED:
        break

    # Start/Stop recording
    if event == "-RECORD-":
        if not recording:
            audio_buffer = np.array([])
            recording = True

            sd.get_stream(callback=audio_callback, channels=1, samplerate=SAMPLE_RATE)
            window["-RECORD-"].update(image_filename="stop_icon.png", image_subsample=2, tooltip="Stop")
            window["-INFO-"].update("Recording...")
        else:
            recording = False
            sd.stream_stop()
            window["-RECORD-"].update(image_filename="record_icon.png", image_subsample=2, tooltip="Record")
            window["-INFO-"].update("Recording stopped.")

            # Perform speaker identification on recorded audio
            if len(audio_buffer) > 0:
                predicted_speaker = identify_speaker(audio_buffer)
                window["-INFO-"].update(f"Is this {predicted_speaker} speaking?")

    # User confirms the speaker
    if event == "-YES-":
        window["-INFO-"].update(f"Welcome {predicted_speaker}!")
    # User denies the speaker
    elif event == "-NO-":
        window["-INFO-"].update("What is your name?")
    # User submits their name
    elif event == "-SUBMIT-":
        user_name = values["-NAME-"]
        window["-INFO-"].update(f"Welcome {user_name}!")

    # Update the live waveform plot
    # waveform_axes = window["-WAVEFORM-"]
    # waveform_axes.erase()
    # if len(audio_buffer) > 0:
    #     waveform_axes.plot(np.linspace(0, RECORD_SECONDS, len(audio_buffer)), audio_buffer, line_color="blue")
    # waveform_axes.draw()

# Close the window and clean up
window.close()

import os
import sys

from PyQt5 import QtGui
from PyQt5.QtWidgets import QApplication, QMainWindow, QListWidgetItem
from main_ui import Ui_MainWindow
from PyQt5.QtCore import QTimer
import threading


AUDIO_SUBFOLDER = "audio"
NOISE_SUBFOLDER = "noise"

# Get the absolute path of the directory containing the current script
PARENT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

# Construct the path to dataset folder
DATASET_ROOT = os.path.join(PARENT_DIRECTORY, 'data\custom')

DATASET_AUDIO_PATH = os.path.join(DATASET_ROOT, AUDIO_SUBFOLDER)
DATASET_NOISE_PATH = os.path.join(DATASET_ROOT, NOISE_SUBFOLDER)

APPLICATION_PATH = os.path.join(PARENT_DIRECTORY, 'app.py')
TRAIN_MODEL_PATH = os.path.join(PARENT_DIRECTORY, 'train_model.py')
PLOT_RECORD_PATH = os.path.join(PARENT_DIRECTORY, 'waveform_fft_output.png')


class SpeakerRecognitionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.path = 'demo'
        self.setup_ui()


        self.refresh_thread = threading.Thread(target=self.refresh_folder_list_threaded)
        self.refresh_thread.daemon = True  # This will allow the thread to exit when the main program exits
        self.refresh_thread.start()

    def setup_ui(self):
        self.refresh_folder_list()  # Call this function to populate the list initially

        # Connect the itemDoubleClicked signal to the slot
        self.ui.folderListWidget.itemDoubleClicked.connect(self.open_selected_folder)

        # Populate the list widget with folder names
        directory = DATASET_AUDIO_PATH  # Replace with your actual directory path
        folder_names = [folder for folder in os.listdir(directory) if os.path.isdir(os.path.join(directory, folder))]
        for folder_name in folder_names:
            item = QListWidgetItem(folder_name)
            self.ui.folderListWidget.addItem(item)


    def refresh_folder_list_threaded(self):
        while True:
            self.refresh_folder_list()
            # Sleep for 5 seconds before the next refresh
            threading.Event().wait(5)

    def refresh_folder_list(self):

        # Clear the existing items in the list widget
        self.ui.folderListWidget.clear()

        # Populate the list widget with updated folder names
        directory = DATASET_AUDIO_PATH
        folder_names = [folder for folder in os.listdir(directory) if os.path.isdir(os.path.join(directory, folder))]
        for folder_name in folder_names:
            item = QListWidgetItem(folder_name)
            self.ui.folderListWidget.addItem(item)

    def open_selected_folder(self, item):
        selected_folder = item.text()
        directory = DATASET_AUDIO_PATH  # Replace with your actual directory path
        folder_path = os.path.join(directory, selected_folder)

        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            os.system(f'explorer "{folder_path}"')  # Opens the folder in the default file explorer


if __name__ == "__main__":
    app = QApplication(sys.argv)

    icon_path = "F:\Documents (Laptop)\SUST\3rd Year 1st Semester\Courses\EEE-332 Digital Signal Processing I Lab\Project\RecMe-VoiceRecognizer\voice-recognition.ico"  # Replace with the actual path to your icon
    app.setWindowIcon(QtGui.QIcon(icon_path))

    window = SpeakerRecognitionApp()
    window.show()
    sys.exit(app.exec_())

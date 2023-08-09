import os
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QListWidgetItem
from main_ui import Ui_MainWindow

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

    def setup_ui(self):

        # Connect the itemDoubleClicked signal to the slot
        self.ui.folderListWidget.itemDoubleClicked.connect(self.open_selected_folder)

        # Populate the list widget with folder names
        directory = DATASET_AUDIO_PATH  # Replace with your actual directory path
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
    window = SpeakerRecognitionApp()
    window.show()
    sys.exit(app.exec_())

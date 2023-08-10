# RecMe-VoiceRecognizer

**RecMe-VoiceRecognizer** is a significant component of the final project for EEE332 Digital Signal Processing Lab 1. The project was conducted under the expert guidance of **Prof. Dr. Md. Raseduzzaman** at **Shahjalal University of Science and Technology**. This voice recognition system aims to showcase the application of digital signal processing techniques in the field of voice analysis and recognition.

The project demonstrates the practical utilization of DSP concepts, and this repository specifically focuses on the voice recognition aspect. The goal is to provide an efficient and accurate system for identifying and distinguishing speakers based on their vocal characteristics.

## Table of Contents

- [Code Organization](#code-organization)
- [Dataset Directory](#dataset-directory)
- [Usage](#usage)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Contact](#contact)

## Code Organization

The project's codebase is organized into different components, each serving a specific purpose. Below is an overview of the key components:

### Training Model

The core training of the model is implemented in the [train_model.py](train_model.py) script. This script is built upon the [Keras Speaker Recognition Example](https://keras.io/examples/audio/speaker_recognition_using_cnn/#introduction), adapting it to our project's requirements. It handles the training and evaluation of the speaker recognition model.

### Console Based Application

The functionality of the application in a console-based environment is housed in the [app.py](app.py) script. This component provides a command-line interface for interacting with the trained model. Users can perform various actions, such as speaker recognition, using this application.

### Graphical User Interface (GUI)

For a more user-friendly interaction, a graphical user interface (GUI) has been developed using the PyQT5 framework. The base code and related `.ui` files are located in the [/pyqt5_ui/](/pyqt5_ui/) directory. The GUI provides an intuitive way for users to interact with the speaker recognition functionality.

- The initial GUI design is in the `.ui` files within [/pyqt5_ui/](/pyqt5_ui/).
- The modified GUI code is implemented in [main_ui.py](main_ui.py), incorporating additional functionality and features.
- The GUI application launcher is [guiapp.py](guiapp.py), which initializes and runs the PyQT5-based GUI application.

This organized structure ensures a clear separation of concerns and makes it easy to locate and modify specific parts of the codebase according to their respective functionalities.



## Dataset Directory

The dataset is stored in the [data](/data) folder. The directory tree is organized as follows:

- `16000_pcm`: This folder contains the downloaded dataset from Kaggle Speaker Recognition, from which only the noise samples are used.

- `custom`: This folder contains the preprocessed data that is used in our application. It is the main dataset folder for the project.

- `raw_data`: This folder contains the raw records of the dataset as they were initially obtained. These raw records are found within the preprocessed data folder.

Please make sure to reference the specific subfolders when working with the dataset in the project.

To access the dataset, you can use the following paths:

- Noise samples: `data/16000_pcm`
- Preprocessed data: `data/custom`
- Raw records: `data/raw_data`

## Usage

### Hardware requirements: 
A working microphone is necessary in order to capture your voice for the application. Make sure the system has detected your microphone.

### Python Libraries: 
A **requirements.txt** file is provided with all the necessary installations for running the application.
Simply run the following code in the parent directory:
> pip install requirements.txt

### Console Application
To use the application in a console, you can run the following code in the project directory terminal
'''python app.py'''
To turn on DEBUG mode, pass the argument -d into hte code as follows
'''
python app.py -d
'''

### GUI Application
To use the GUI Application, run the following code in the parent directory:
'''
python guiapp.py
'''

### Builds
You can also access the [builds](\builds) folder to install the application using the **setup.exe**, which will automatically install necessary components to run the application, and launch the application using **RecMe.exe** in the installation directory.

## License

Anyone is free to use this application under the MIT license. You can credit me if you want. :)

## Acknowledgments

Keras Speaker Recognition Example was the core behind this project. Do send it some love.

## Contact

You can contact me using my email: irfannafizislive@gmail.com

# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'test.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setWindowModality(QtCore.Qt.WindowModal)
        MainWindow.resize(921, 565)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("C:/Users/irfan/.designer/backup/icons/voice-recognition.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        MainWindow.setWindowOpacity(4.0)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        spacerItem = QtWidgets.QSpacerItem(45, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem, 1, 2, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(20, 39, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout.addItem(spacerItem1, 2, 1, 1, 1)
        spacerItem2 = QtWidgets.QSpacerItem(46, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem2, 1, 0, 1, 1)
        spacerItem3 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout.addItem(spacerItem3, 0, 1, 1, 1)
        self.verticalLayout_11 = QtWidgets.QVBoxLayout()
        self.verticalLayout_11.setObjectName("verticalLayout_11")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.labelAppTitle = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("RomanD")
        font.setPointSize(16)
        self.labelAppTitle.setFont(font)
        self.labelAppTitle.setAlignment(QtCore.Qt.AlignCenter)
        self.labelAppTitle.setObjectName("labelAppTitle")
        self.verticalLayout_3.addWidget(self.labelAppTitle)
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.verticalLayout_3.addWidget(self.line)
        self.verticalLayout_11.addLayout(self.verticalLayout_3)
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.verticalLayout_9 = QtWidgets.QVBoxLayout()
        self.verticalLayout_9.setObjectName("verticalLayout_9")
        self.folderLabel = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Rockwell")
        font.setPointSize(12)
        self.folderLabel.setFont(font)
        self.folderLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.folderLabel.setObjectName("folderLabel")

        self.verticalLayout_9.addWidget(self.folderLabel)

        self.folderListWidget = QtWidgets.QListWidget(self.centralwidget)
        self.folderListWidget.setObjectName("folderListWidget")

        self.verticalLayout_9.addWidget(self.folderListWidget)
        self.horizontalLayout_9.addLayout(self.verticalLayout_9)
        self.verticalLayout_10 = QtWidgets.QVBoxLayout()
        self.verticalLayout_10.setObjectName("verticalLayout_10")
        self.viewerLabel = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Rockwell")
        font.setPointSize(12)
        self.viewerLabel.setFont(font)
        self.viewerLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.viewerLabel.setObjectName("viewerLabel")
        self.verticalLayout_10.addWidget(self.viewerLabel)
        self.verticalLayout_7 = QtWidgets.QVBoxLayout()
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout()
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.waveformViewer = QtWidgets.QTextEdit(self.centralwidget)
        self.waveformViewer.setObjectName("waveformViewer")
        self.verticalLayout_6.addWidget(self.waveformViewer)
        self.recordButton = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Rockwell")
        self.recordButton.setFont(font)
        self.recordButton.setObjectName("recordButton")
        self.verticalLayout_6.addWidget(self.recordButton)
        self.retrainButton = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Rockwell")
        self.retrainButton.setFont(font)
        self.retrainButton.setObjectName("retrainButton")
        self.verticalLayout_6.addWidget(self.retrainButton)
        self.addButton = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Rockwell")
        self.addButton.setFont(font)
        self.addButton.setObjectName("addButton")
        self.verticalLayout_6.addWidget(self.addButton)
        self.browseButton = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Rockwell")
        self.browseButton.setFont(font)
        self.browseButton.setObjectName("browseButton")
        self.verticalLayout_6.addWidget(self.browseButton)
        self.quitButton = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Rockwell")
        self.quitButton.setFont(font)
        self.quitButton.setObjectName("quitButton")
        self.verticalLayout_6.addWidget(self.quitButton)
        self.verticalLayout_7.addLayout(self.verticalLayout_6)
        self.verticalLayout_10.addLayout(self.verticalLayout_7)
        self.horizontalLayout_9.addLayout(self.verticalLayout_10)
        self.verticalLayout_8 = QtWidgets.QVBoxLayout()
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.consoleLabel = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Rockwell")
        font.setPointSize(12)
        self.consoleLabel.setFont(font)
        self.consoleLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.consoleLabel.setObjectName("consoleLabel")
        self.verticalLayout_8.addWidget(self.consoleLabel)
        self.consoleBrowser = QtWidgets.QTextBrowser(self.centralwidget)
        self.consoleBrowser.setObjectName("consoleBrowser")
        self.verticalLayout_8.addWidget(self.consoleBrowser)
        self.horizontalLayout_9.addLayout(self.verticalLayout_8)
        self.verticalLayout_11.addLayout(self.horizontalLayout_9)
        self.gridLayout.addLayout(self.verticalLayout_11, 1, 1, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 921, 22))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuEdit = QtWidgets.QMenu(self.menubar)
        self.menuEdit.setObjectName("menuEdit")
        self.menuView = QtWidgets.QMenu(self.menubar)
        self.menuView.setObjectName("menuView")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionRecord = QtWidgets.QAction(MainWindow)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("C:/Users/irfan/.designer/backup/icons/record_icon.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionRecord.setIcon(icon1)
        self.actionRecord.setObjectName("actionRecord")
        self.actionRe_Train = QtWidgets.QAction(MainWindow)
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap("C:/Users/irfan/.designer/backup/icons/retrain_icon.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionRe_Train.setIcon(icon2)
        self.actionRe_Train.setObjectName("actionRe_Train")
        self.actionAdd_Dataset = QtWidgets.QAction(MainWindow)
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap("C:/Users/irfan/.designer/backup/icons/add_dataset_icon.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionAdd_Dataset.setIcon(icon3)
        self.actionAdd_Dataset.setObjectName("actionAdd_Dataset")
        self.actionQuit = QtWidgets.QAction(MainWindow)
        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap("C:/Users/irfan/.designer/backup/icons/quit_icon.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionQuit.setIcon(icon4)
        self.actionQuit.setObjectName("actionQuit")
        self.actionBrowse_Dataset = QtWidgets.QAction(MainWindow)
        icon5 = QtGui.QIcon()
        icon5.addPixmap(QtGui.QPixmap("C:/Users/irfan/.designer/backup/icons/folder_icon.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionBrowse_Dataset.setIcon(icon5)
        self.actionBrowse_Dataset.setObjectName("actionBrowse_Dataset")
        self.actionBrowse_Raw_Dataset = QtWidgets.QAction(MainWindow)
        self.actionBrowse_Raw_Dataset.setIcon(icon5)
        self.actionBrowse_Raw_Dataset.setObjectName("actionBrowse_Raw_Dataset")
        self.actionBrowse_Noise_Dataset = QtWidgets.QAction(MainWindow)
        self.actionBrowse_Noise_Dataset.setIcon(icon5)
        self.actionBrowse_Noise_Dataset.setObjectName("actionBrowse_Noise_Dataset")
        self.actionNormal = QtWidgets.QAction(MainWindow)
        self.actionNormal.setObjectName("actionNormal")
        self.actionMaximize = QtWidgets.QAction(MainWindow)
        self.actionMaximize.setObjectName("actionMaximize")
        self.menuFile.addAction(self.actionRecord)
        self.menuFile.addAction(self.actionRe_Train)
        self.menuFile.addAction(self.actionAdd_Dataset)
        self.menuFile.addAction(self.actionQuit)
        self.menuEdit.addAction(self.actionBrowse_Dataset)
        self.menuEdit.addAction(self.actionBrowse_Raw_Dataset)
        self.menuEdit.addAction(self.actionBrowse_Noise_Dataset)
        self.menuView.addAction(self.actionNormal)
        self.menuView.addAction(self.actionMaximize)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuEdit.menuAction())
        self.menubar.addAction(self.menuView.menuAction())

        self.retranslateUi(MainWindow)
        self.actionQuit.triggered.connect(MainWindow.close) # type: ignore
        self.actionMaximize.triggered.connect(MainWindow.showMaximized) # type: ignore
        self.actionNormal.triggered.connect(MainWindow.showNormal) # type: ignore
        self.quitButton.clicked.connect(MainWindow.close) # type: ignore
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "RecMe - The Speaker Identifier"))
        self.labelAppTitle.setText(_translate("MainWindow", "RecMe - The Speaker Identifier"))
        self.folderLabel.setText(_translate("MainWindow", "People in Dataset"))
        self.viewerLabel.setText(_translate("MainWindow", "Waveform Viewier"))
        self.recordButton.setStatusTip(_translate("MainWindow", "Press to start recording an audio for identification"))
        self.recordButton.setText(_translate("MainWindow", "Record"))
        self.retrainButton.setStatusTip(_translate("MainWindow", "Press to re-train the machine learning model"))
        self.retrainButton.setText(_translate("MainWindow", "Re-Train Model"))
        self.addButton.setStatusTip(_translate("MainWindow", "Press to add a new person to the dataset saying the passphrase 10 times"))
        self.addButton.setText(_translate("MainWindow", "Add Person"))
        self.browseButton.setStatusTip(_translate("MainWindow", "Open the root dataset directory"))
        self.browseButton.setText(_translate("MainWindow", "Browse Dataset"))
        self.quitButton.setStatusTip(_translate("MainWindow", "Press to quit the application"))
        self.quitButton.setText(_translate("MainWindow", "Quit"))
        self.consoleLabel.setText(_translate("MainWindow", "Console Output"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.menuEdit.setTitle(_translate("MainWindow", "Edit"))
        self.menuView.setTitle(_translate("MainWindow", "View"))
        self.actionRecord.setText(_translate("MainWindow", "Record"))
        self.actionRecord.setStatusTip(_translate("MainWindow", "Record an audio for identification"))
        self.actionRecord.setShortcut(_translate("MainWindow", "Ctrl+R"))
        self.actionRe_Train.setText(_translate("MainWindow", "Re-Train"))
        self.actionRe_Train.setStatusTip(_translate("MainWindow", "Re-train the machine learning model"))
        self.actionAdd_Dataset.setText(_translate("MainWindow", "Add Dataset"))
        self.actionAdd_Dataset.setStatusTip(_translate("MainWindow", "Add new person saying the passphrase 10 times"))
        self.actionAdd_Dataset.setShortcut(_translate("MainWindow", "Ctrl+Shift+N"))
        self.actionQuit.setText(_translate("MainWindow", "Quit"))
        self.actionQuit.setStatusTip(_translate("MainWindow", "Quit the application"))
        self.actionQuit.setShortcut(_translate("MainWindow", "Esc"))
        self.actionBrowse_Dataset.setText(_translate("MainWindow", "Browse Dataset "))
        self.actionBrowse_Dataset.setStatusTip(_translate("MainWindow", "Open the directory of processed dataset"))
        self.actionBrowse_Dataset.setShortcut(_translate("MainWindow", "Ctrl+B"))
        self.actionBrowse_Raw_Dataset.setText(_translate("MainWindow", "Browse Raw Dataset"))
        self.actionBrowse_Raw_Dataset.setStatusTip(_translate("MainWindow", "Open the directory of raw dataset"))
        self.actionBrowse_Noise_Dataset.setText(_translate("MainWindow", "Browse Noise Dataset"))
        self.actionBrowse_Noise_Dataset.setStatusTip(_translate("MainWindow", "Open the directory of noise dataset"))
        self.actionNormal.setText(_translate("MainWindow", "Normal"))
        self.actionNormal.setStatusTip(_translate("MainWindow", "Resize the application to normal view"))
        self.actionMaximize.setText(_translate("MainWindow", "Maximize"))
        self.actionMaximize.setStatusTip(_translate("MainWindow", "Maximize the application window"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

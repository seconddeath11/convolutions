import itertools
import sys
import time

import cv2
import numpy as np
from PIL import Image
from PyQt5 import QtCore
from PyQt5.QtCore import Qt, QModelIndex
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QMessageBox, QMenuBar, QAction

import design, filters


def timeit(f):
    def timed(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()

        print('func:%r args:[%r, %r] took: %2.4f sec' % (f.__name__, args, kw, te - ts))
        return result

    return timed


class TableModel(QtCore.QAbstractTableModel):
    def __init__(self, data):
        super(TableModel, self).__init__()
        self._data = data

    def setData(self, index: QModelIndex, value, role: int = ...) -> bool:
        if role == Qt.EditRole:
            if value == '':
                self._data[index.row()][index.column()] = 0
            else:
                self._data[index.row()][index.column()] = value
            return True

    def get_data(self):
        return self._data

    def update(self, data):
        self.beginResetModel()
        self._data = data
        self.endResetModel()

    def rowCount(self, parent=QtCore.QModelIndex()):
        return len(self._data)

    def columnCount(self, parent=QtCore.QModelIndex()):
        return len(self._data[0])

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if role == Qt.DisplayRole:
            return self._data[index.row()][index.column()]

    def flags(self, index):
        return Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsEditable


class FilterApp(QMainWindow, design.Ui_MainWindow):
    # TODO: filter_size
    def __init__(self):
        super().__init__()
        self.mode = QImage.Format_RGB888 #if self.channels_number == 3 else QImage.Format_Grayscale8
        self.channels_number = 3 #if len(self.data.shape) > 2 else 1
        self.setupUi(self)
        self.filename = None
        self.data = None

        self.startButton.clicked.connect(self.set_output)
        self.setStyleSheet("QTableView {background: #f0f0f0;}")
        data = [
            [-1, -1, -1],
            [-1, 9, -1],
            [-1, -1, -1]
        ]
        self.model = TableModel(data)
        self.table.setModel(self.model)
        self.comboBox.addItems(["Custom", "H1", "H2", "H3", "median", "H4", "H5", "H6"])
        self.comboBox.currentTextChanged.connect(self.on_combobox_changed)
        self.openAction.triggered.connect(self.set_input)
        self.exitAction.triggered.connect(exit)

    def open_file(self):
        file_dialog = QFileDialog()
        file_dialog.setAcceptMode(QFileDialog.AcceptOpen)
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        filename = file_dialog.getOpenFileName(self, 'Open File', '', "Images (*.png *.bmp *.pcx *.jpg)")
        self.filename = filename[0]

    def set_input(self):
        self.open_file()
        if self.filename:
            try:
                with Image.open(self.filename) as img:
                    self.data = np.asarray(img.convert('RGB')).astype("uint8")
            except FileNotFoundError:
                self.show_message("File not found")
                return
            except Exception:
                self.show_message("Something went wrong")
                return

            self.set_image(self.data, self.inputLabel)

    # TODO: nulls message
    def set_output(self):
        try:
            kernel = np.array(self.table.model().get_data()).astype("float32")
        except ValueError:
            self.show_message("The kernel is bad")
            return
        if self.data is None:
            self.show_message("Please, choose an image first")
            return
        image_filter = Filter(self.channels_number)
        image_filter.image = self.data
        kernel_mode = self.comboBox.currentText()
        if kernel_mode == "median":
            out = image_filter.filter(kernel=np.ones((3, 3)), function=np.median)
        else:
            out = image_filter.filter(kernel=kernel)

        self.set_image(out, self.outputLabel)

    def set_image(self, data, label):
        try:
            image = QImage(data, data.shape[1], data.shape[0], self.channels_number * data.shape[1], self.mode)
        except Exception:
            self.show_message("Couldn't create output")
            return
        try:
            pixmap = QPixmap(image).scaled(label.width(), label.height(), QtCore.Qt.KeepAspectRatio)
            label.setPixmap(pixmap)
        except Exception:
            self.show_message("Couldn't set output")

    def show_message(self, message, title="Error!"):
        dlg = QMessageBox(self)
        dlg.setWindowTitle(title)
        dlg.setText(message)
        dlg.exec()

    @staticmethod
    def choose_kernel(text):
        return {
            "H1": filters.h1,
            "H2": filters.h2,
            "H3": filters.h3,
            "H4": filters.h4,
            "H5": filters.h5,
            "H6": filters.h6
        }.get(text)

    def on_combobox_changed(self, value):
        kernel = self.choose_kernel(value)
        if kernel is not None:
            self.table.model().update(kernel.tolist())


class Filter:
    def __init__(self, depth):
        self._image = None
        self.depth = depth

    @property
    def image(self):
        return self._image

    @image.setter
    def image(self, value):
        self._image = value

    @timeit
    def filter(self, kernel=filters.h1, function=np.sum):
        output = []
        inp = cv2.split(self.image)
        for i in range(self.depth):
            output.append(np.array(self.filter_channel(inp[i], kernel, function)))
        output = np.moveaxis(np.array(output), 0, -1).copy()
        output[output < 0] = 0
        output[output > 255] = 255
        return output.astype('uint8')

    def filter_channel(self, channel, kernel, function=np.sum):
        img = self.im2col(channel, kernel.shape[:2])
        output = np.zeros(img.shape[:2])
        kernel_flat = kernel.flatten()
        for x, y in itertools.product(range(img.shape[0]),
                                      range(img.shape[1])):
            output[x, y] = function(img[x, y] * kernel_flat)
        return output

    @staticmethod
    def im2col(img, kernel_size):
        return np.lib.stride_tricks.sliding_window_view(img, kernel_size).reshape(
            img.shape[0] - int(kernel_size[0] / 2) * 2, img.shape[1]
            - int(kernel_size[1] / 2) * 2, -1)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = FilterApp()
    window.show()
    app.exec_()

from qt import Ui_MainWindow
from PyQt5 import QtCore, QtGui, QtWidgets
from PIL import Image, ImageQt
import sys


class my_window(QtWidgets.QMainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.button_open = self.ui.button_open
        self.button_front = self.ui.button_front
        self.button_back = self.ui.button_back
        self.button_profile = self.ui.button_profile
        self.button_all = self.ui.button_all
        self.textBrowser = self.ui.textBrowser
        self.label_1 = self.ui.label_1
        self.label_2 = self.ui.label_2
        self.centralWidget = self.ui.centralwidget
        self.button_open.clicked.connect(self.open_img)
        self.button_front.clicked.connect(self.front)
        self.button_back.clicked.connect(self.back)
        self.button_profile.clicked.connect(self.profile)
        self.button_all.clicked.connect(self.all)

    def open_img(self):
        img_name, img_type = QtWidgets.QFileDialog.getOpenFileName(self.centralWidget, "打开图片", "",
                                                                   "*.jpg;;*.jpeg;;*.png;;All Files(*)")
        self.photo = Image.open(img_name)
        # w, h = self.photo.size
        w, h = self.resize_photo(self.label_1.width() - 1, self.label_1.height() - 1, self.photo)
        image_qt = ImageQt.ImageQt(self.photo)
        mp = QtGui.QPixmap.fromImage(image_qt)
        mp = mp.scaled(w, h)
        self.label_1.setPixmap(mp)

    def resize_photo(self, w_box, h_box, pil_image):
        w, h = pil_image.size  # 获取图像的原始大小
        f1 = 1.0 * w_box / w
        f2 = 1.0 * h_box / h
        factor = min([f1, f2])
        width = int(w * factor)
        height = int(h * factor)
        return width, height

    def front(self):
        pass

    def back(self):
        pass

    def profile(self):
        pass

    def all(self):
        pass


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = my_window()
    window.show()
    sys.exit(app.exec_())

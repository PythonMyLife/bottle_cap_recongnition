# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'qt.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1000, 800)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label_1 = QtWidgets.QLabel(self.centralwidget)
        self.label_1.setGeometry(QtCore.QRect(240, 0, 751, 400))
        self.label_1.setText("")
        self.label_1.setObjectName("label_1")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(240, 400, 751, 400))
        self.label_2.setText("")
        self.label_2.setObjectName("label_2")
        self.textBrowser = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser.setGeometry(QtCore.QRect(0, 70, 231, 151))
        self.textBrowser.setObjectName("textBrowser")
        self.button_front = QtWidgets.QPushButton(self.centralwidget)
        self.button_front.setGeometry(QtCore.QRect(60, 230, 101, 41))
        self.button_front.setObjectName("button_front")
        self.button_profile = QtWidgets.QPushButton(self.centralwidget)
        self.button_profile.setGeometry(QtCore.QRect(60, 300, 101, 41))
        self.button_profile.setObjectName("button_profile")
        self.button_back = QtWidgets.QPushButton(self.centralwidget)
        self.button_back.setGeometry(QtCore.QRect(60, 380, 101, 41))
        self.button_back.setObjectName("button_back")
        self.button_all = QtWidgets.QPushButton(self.centralwidget)
        self.button_all.setGeometry(QtCore.QRect(60, 450, 101, 41))
        self.button_all.setObjectName("button_all")
        self.button_open = QtWidgets.QPushButton(self.centralwidget)
        self.button_open.setGeometry(QtCore.QRect(60, 10, 101, 41))
        self.button_open.setObjectName("button_open")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionopen = QtWidgets.QAction(MainWindow)
        self.actionopen.setObjectName("actionopen")

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.button_front.setText(_translate("MainWindow", "正面检测"))
        self.button_profile.setText(_translate("MainWindow", "侧面检测"))
        self.button_back.setText(_translate("MainWindow", "反面检测"))
        self.button_all.setText(_translate("MainWindow", "形态检测"))
        self.button_open.setText(_translate("MainWindow", "打开图片"))
        self.actionopen.setText(_translate("MainWindow", "open"))

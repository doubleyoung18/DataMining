#!/usr/bin/env/python
# -*- coding:utf-8 -*-

"""
@Author:Double Young
@Time:2018/12/04 15:27:19
@Desc:登录界面
"""

from PyQt5.QtWidgets import QApplication, QMainWindow, \
    QDesktopWidget, QLabel, QLineEdit, QPushButton
import sys

class LoginFrame(QMainWindow):
    def __init__(self):
        """
        构造登录窗口
        """
        super().__init__()
        self.initUI()

    def initUI(self):
        """
        初始化界面
        :return: 空
        """
        self.setWindowTitle("PyQQ登录")  # 设置窗口标题
        self.setGeometry(0, 0, 400, 300)  # 设置窗口大小(x,y,w,h)
        self.center()  # 设置窗口居中

        backgroundLabel = QLabel()  # 背景标签

        profileLabel = QLabel()  # 头像标签

        idEdit = QLineEdit(self)  # 账号输入框
        idEdit.setPlaceholderText("请输入账号")
        idEdit.setGeometry(100, 60, 200, 30)

        pwdEdit = QLineEdit(self)  # 密码输入框
        pwdEdit.setPlaceholderText("请输入密码")
        pwdEdit.setGeometry(100, 100, 200, 30)

        self.loginButton = QPushButton("登录", self)
        self.loginButton.setGeometry(100, 200, 200, 30)
        # self.loginButton.addAction("登录")
        self.show()  # 显示窗口

    def center(self):
        """
        设置窗口居中
        :return: 空
        """
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def Action(self):
        if self.loginButton.isEnabled():
            pass


if __name__ == "__main__":
    app = QApplication(sys.argv)
    test = LoginFrame()
    sys.exit(app.exec_())



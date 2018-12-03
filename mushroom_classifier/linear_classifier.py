#!/usr/bin/env/python
# -*- coding:utf-8 -*-

"""
@Author:Double Young
@Time:2018/12/02 16:50:50
@Desc:线性分类器，对蘑菇数据集通过梯度下降和随机梯度下降分类
"""

import numpy as np  # 科学计算
import pandas as pd  # 数据读入
import matplotlib.pyplot as plt  # 画图
from sklearn import preprocessing  # 数据预处理
from sklearn.model_selection import train_test_split  # 数据切分

def load_data(filename):
    """
    数据读入
    :param filename: 数据集文件名
    :return: 读入的数据集
    """
    # 属性列表
    attributes_name = ["class", "cap-shape", "cap-surface",
                       "cap-color", "bruises", "odor",
                       "gill-attachment", "gill-spacing",
                       "gill-size", "gill-color", "stalk-shape",
                       "stalk-root", "stalk-surface-above-ring",
                       "stalk-surface-below-ring",
                       "stalk-color-above-ring",
                       "stalk-color-below-ring", "veil-type",
                       "veil-color", "ring-number", "ring-type",
                       "spore-print-color", "population", "habitat"]
    # 从文件中读入
    data = pd.read_csv(filename, header=None, names=attributes_name)
    return data

def preprocess_data(data):
    """
    数据预处理（对x离散+规范，对y缩放）
    :param data: 数据集
    :return: 预处理后的数据集
    """
    x = data.iloc[1:, 1:]  # 分离x
    y = data.iloc[1:, 0].values  # 分离y
    x = pd.get_dummies(x)  # 对x离散化
    ss = preprocessing.StandardScaler()  # 对x规范化 x=(x-mean)/std
    x = ss.fit_transform(x)
    y = np.where(y == 'e', -1, 1).reshape(-1, 1)  # 对y缩放到[-1,1]
    data = np.hstack((y, x))
    return data

def split_data(data, test_percent):
    """
    数据划分
    :param data: 预处理后的数据集
    :param test_percent: 测试集比例
    :return: 训练集输入，训练集输出，测试集输入，测试集输出
    """
    # 划分训练集和测试集
    (train, test) = train_test_split(data, test_size=test_percent)
    # 划分输入和输出
    x_train = train[:, 1:]
    y_train = train[:, 0]
    x_test = test[:, 1:]
    y_test = test[:, 0]
    return x_train, y_train, x_test, y_test

def cal_h(w, x):
    """
    计算预测函数
    :param w: 参数向量
    :param x: 输入向量
    :return: 预测函数值
    """
    return np.matmul(w, np.transpose(x))  # h=wx=w1x1+w2x2+...wnxn+b

def gd_classifier(x, y, alpha, max_iter, tol):
    """
    梯度下降训练线性分类器
    :param x: 测试集输入
    :param y: 测试集输出
    :param alpha: 学习率
    :param max_iter: 迭代门限值
    :param tol: 误差门限值
    :return: 训练后参数
    """
    m = len(x)  # 数据项总数
    x = np.hstack((x, np.ones((m, 1))))  # x增加一列全1
    n = len(x[0])  # 特征总数（含新增列）
    w = np.random.random(n)  # 系数向量
    count = 0  # 迭代记录
    loss_list = []  # 误差记录
    plt.figure(num=1)
    plt.title("loss of gradient descent")
    while count < max_iter:  # 若大于迭代门限值，则结束迭代
        diff = np.zeros(n)  # 偏导
        loss = 0  # 误差值
        for i in range(m):
            h = cal_h(w, x[i])  # 预测值
            diff += (h - y[i]) * x[i]  # 求偏导
            loss += (h - y[i]) ** 2 / m  # 求误差
        w -= alpha * diff / m  # 更新系数
        # 判断是否收敛
        if count != 0 and abs(loss-loss_list[count-1]) < tol:
            break
        count += 1
        loss_list.append(loss)
        if count % 10 == 0:
            print("第{0}次迭代误差:{1}".format(count, loss))
    # 绘制误差函数
    plt.plot(range(count), loss_list)
    plt.xlabel("iter")
    plt.ylabel("loss")
    plt.draw()
    return w

def sgd_classifier(x, y, alpha, max_iter, tol):
    """
    随机梯度下降训练线性分类器
    :param x: 测试集输入
    :param y: 测试集输出
    :param alpha: 学习率
    :param max_iter: 迭代门限值
    :param tol: 误差门限值
    :return: 训练后参数
    """
    m = len(x)  # 数据项总数
    x = np.hstack((x, np.ones((m, 1))))  # x增加一列全1
    n = len(x[0])  # 特征总数（含新增列）
    w = np.random.random(n)  # 系数向量
    count = 0  # 迭代记录
    loss_list = []  # 误差记录
    plt.figure(num=2)
    plt.title("loss of stochastic gradient descent")
    while count < max_iter:  # 若大于迭代门限值，则结束迭代
        diff = np.zeros(n)  # 偏导
        loss = 0  # 误差值
        i = np.random.randint(0, m)
        h = cal_h(w, x[i])  # 预测值
        diff += (h - y[i]) * x[i]  # 求偏导
        loss += (h - y[i]) ** 2 / m  # 求误差
        w -= alpha * diff / m  # 更新系数
        # 判断是否收敛
        if count != 0 and abs(loss-loss_list[count-1]) < tol:
            break
        count += 1
        loss_list.append(loss)
        if count % 10000 == 0:
            print("第{0}次迭代误差:{1}".format(count, loss))
    # 绘制误差函数
    plt.plot(range(count), loss_list)
    plt.xlabel("iter")
    plt.ylabel("loss")
    plt.draw()
    return w

def test(x, y, w):
    """
    测试
    :param x: 测试集输入
    :param y: 测试集输出
    :param w: 训练后参数
    :return: 准确度，错误数
    """
    m = len(x)  # 测试集总数
    n = 0  # 正确预测总数
    x = np.hstack((x, np.ones((m, 1))))  # 测试输入加上一列全1
    for i in range(m):  # 测试
        h = cal_h(w, x[i])
        # print("预测：", h, "真实：", y[i])
        if h * y[i] > 0:
            n += 1
    accuracy = n / m  # 准确度
    return accuracy, (m-n)

if __name__ == "__main__":
    filename = "agaricus-lepiota.data.txt"
    # 数据读入
    data = load_data(filename)
    print("读入数据:\n{0}".format(data.head(5)))
    print("-------------------------------------------------")

    # 数据预处理
    data = preprocess_data(data)
    print("特征离散和特征规范化后数据:\n{0}".format(data))
    print("-------------------------------------------------")

    # 梯度下降分类
    test_percent = 0.5
    alpha = 0.1
    max_iter = 50
    # 数据切分
    x_train, y_train, x_test, y_test = split_data(data, test_percent)
    # 梯度下降训练
    w = gd_classifier(x_train, y_train, alpha, max_iter, 1e-20)
    # 梯度下降测试测试
    accuracy, n = test(x_test, y_test, w)
    print("选取{0}%测试集,学习率:{1},迭代{2}次"
          .format(test_percent*100, alpha, max_iter))
    print("梯度下降准确率:{0},错误数:{1}".format(accuracy,n))
    print("-------------------------------------------------")

    # 随机梯度下降分类
    test_percent = 0.5
    alpha = 0.1
    max_iter = 100000
    # 数据切分
    x_train, y_train, x_test, y_test = split_data(data, test_percent)
    # 随机梯度下降训练
    w = sgd_classifier(x_train, y_train, alpha, max_iter, 1e-20)
    # 随机梯度下降测试
    accuracy, n = test(x_test, y_test, w)
    print("选取{0}%测试集,学习率:{1},迭代:{2}次"
          .format(test_percent * 100, alpha, max_iter))
    print("随机梯度下降准确率:{0},错误数:{1}".format(accuracy, n))

    # 显示图像
    plt.show()
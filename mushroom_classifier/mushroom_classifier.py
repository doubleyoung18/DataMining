#!/usr/bin/env/python
# -*- coding:utf-8 -*-

"""
@Author:Double Young
@Time:2018/11/19 18:28:33
"""

# 数据处理
import numpy as np
import pandas as pd
from sklearn import preprocessing  # 数据规范化
from sklearn.model_selection import train_test_split  # 划分训练和测试集
from sklearn.linear_model import LogisticRegression  # 逻辑回归模型
from sklearn.metrics import accuracy_score  # 准确度计算

# 属性列表
attributes_name = ["class", "cap-shape", "cap-surface", "cap-color",
                    "bruises", "odor", "gill-attachment",
                    "gill-spacing", "gill-size", "gill-color",
                    "stalk-shape", "stalk-root",
                    "stalk-surface-above-ring",
                    "stalk-surface-below-ring",
                    "stalk-color-above-ring",
                    "stalk-color-below-ring", "veil-type",
                    "veil-color", "ring-number", "ring-type",
                    "spore-print-color", "population", "habitat"]

def load_data(filename):
    """
    数据预处理
    :param filename: 数据集文件名
    :return: 预处理后的数据集
    """
    # 读取数据
    data = pd.read_csv(filename, header=None, names=attributes_name)
    # 数据规范化
    encoder = preprocessing.LabelEncoder()
    for col in data.columns:
        data[col] = encoder.fit_transform(data[col])
    return data

def split_data(data):
    """
    划分训练集和结果集
    :param data: 预处理后的数据集
    :return: 训练集输入，训练集输出，测试集输入，测试集输出
    """
    (train, test) = train_test_split(data, test_size=0.5)  # 1:1
    # 划分输入输出
    x_train = train[[x for x in train if "class" not in x]]
    y_train = train["class"]
    x_test = test[[x for x in test if "class" not in x]]
    y_test = test["class"]
    return x_train, y_train, x_test, y_test

def logistic_regression(x_train, y_train, x_test):
    """
    利用逻辑回归预测结果
    :param x_train: 训练集输入
    :param y_train: 训练集输出
    :param x_test: 测试集输入
    :return: 预测输出
    """
    lr = LogisticRegression()
    lr.fit(x_train, y_train)
    y_predict = lr.predict(x_test)
    return y_predict

def cal_accuracy(y_test, y_predict):
    """
    计算预测准确度
    :param y_test: 测试集输出
    :param y_predict: 预测输出
    :return: 准确率
    """
    accuracy = accuracy_score(y_test, y_predict)
    return accuracy

if __name__ == "__main__":
    filename = "mushroom_data.txt"
    data = load_data(filename)
    x_train, y_train, x_test, y_test = split_data(data)
    y_predict = logistic_regression(x_train, y_train, x_test)
    accuracy = cal_accuracy(y_test, y_predict)
    print("Accuracy of Logistic Regression is %f" % accuracy)

    sample = data.sample(n=1, axis=0)
    x_sample = sample[[x for x in sample if "class" not in x]]
    y_sample = sample["class"]
    y_predict = logistic_regression(x_train, y_train, x_sample)
    print("Sample's real class:", y_sample.iloc[0])
    print("Sample's predict class:", y_predict[0])

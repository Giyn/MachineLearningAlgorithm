# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 20:08:04 2020

@author: Giyn
"""

import csv
from random import randrange


def read_csv(filename):
    """
    读取数据集csv文件
    Args:
            filename: 文件路径
    Returns:
            dataSet: 数据集
    """
    dataSet = []
    with open(filename, 'r') as f:
        csvReader = csv.reader(f) # 读取每一行，返回list
        for line in csvReader:
            dataSet.append(line)
    return dataSet


def feature_to_float(dataSet):
    """
    将特征转换为float类型
    Args:
            dataSet: 数据集
    Returns: None
    """
    featLen = len(dataSet[0]) - 1 # 除去target那一列
    for data in dataSet:
        for column in range(featLen):
            pass
            # data[column] = float(data[column].strip())


def spilt_dataSet(dataSet, n_folds):
    """
    将数据集分成n份，用于交叉验证
    Args:
            dataSet: 数据集
            n_folds: 数据集分割份数
    Returns: 分割后的数据集
    """
    fold_size = int(len(dataSet) / n_folds)
    dataSet_copy = list(dataSet)
    dataSet_spilt = []
    for i in range(n_folds):
        fold = []
        while len(fold) < fold_size:
            index = randrange(len(dataSet_copy)) # 返回序列中随机一个数
            fold.append(dataSet_copy.pop(index)) # pop函数弹出list最后一个值并返回
        dataSet_spilt.append(fold)
    return dataSet_spilt


def data_spilt(dataSet, index, value):
    """
    分割数据集
    Args:
            dataSet: 数据集
            index: 数据集分割索引
            value: 分割阈值
    Returns
            left: 数据集的一部分
            right: 数据集的另一部分
    """
    left = []
    right = []
    for row in dataSet:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right
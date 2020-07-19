# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 20:08:04 2020

@author: Giyn
"""

from random import seed
from RandomForest import RandomForest
from data_handle import read_csv, feature_to_float, spilt_dataSet


if __name__ == '__main__':
    seed(1)
    dataSet = read_csv('sonar-all-data.csv') # 读取数据
    feature_to_float(dataSet) # 转换特征值
    n_folds = 5 # 分割交叉验证集
    max_depth = 15 # 树的深度
    min_size = 1 # 停止的分枝样本最小数目
    ratio = 1.0 # 随机采样的比例
    n_features = 15 # 选取特征
    n_trees = 10 # 决策树数量
    folds = spilt_dataSet(dataSet, n_folds)
    scores = []
    alg = RandomForest()
    for fold in folds:
        train_set = folds[:] # 拷贝
        train_set.remove(fold)
        train_set = sum(train_set, []) # 将多个fold列表组合成一个train_set列表
        test_set = []
        for row in fold:
            row_copy = list(row)
            row_copy[-1] = None
            test_set.append(row_copy)
        actual = [row[-1] for row in fold]
        predict_values = alg.RF(train_set, test_set, ratio, n_features, max_depth, min_size, n_trees)
        accur = alg.accuracy(predict_values, actual)
        scores.append(accur)

    print ('准确率:%s' % scores)
    print ('平均准确率:', "{:2%}".format((sum(scores) / float(len(scores)))))